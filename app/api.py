from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.auth import (
    AuthContext,
    authenticate_credentials,
    clear_session_cookies,
    create_session,
    create_user,
    require_auth,
    require_csrf,
    revoke_session,
    set_session_cookies,
)
from app.chat import ChatService
from app.config import Settings, get_settings
from app.database import get_db
from app.embeddings import get_embedding_provider
from app.llm import get_model_provider
from app.models import Conversation, User
from app.rag import Retriever
from app.rate_limit import RateLimitExceeded, get_rate_limiter
from app.schemas import (
    AccountDeleteRequest,
    ChatRequest,
    ChatResponse,
    LoginRequest,
    RegisterRequest,
    UserOut,
)
from app.security import verify_password

router = APIRouter(prefix="/api")


def _client_key(request: Request, action: str) -> str:
    host = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    return f"{forwarded or host}:{action}"


def _limit(request: Request, action: str, limit: int, seconds: int) -> None:
    try:
        get_rate_limiter().check(_client_key(request, action), limit, seconds)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429, detail="Too many requests; please try again later"
        ) from exc


def _user_out(user: User, settings: Settings) -> UserOut:
    return UserOut(id=user.id, email=user.email, history_enabled=settings.save_chat_history)


@router.post("/auth/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UserOut:
    _limit(request, "register", 5, 3600)
    if not settings.allow_registration:
        raise HTTPException(status_code=403, detail="Registration is currently closed")
    user = create_user(db, str(payload.email), payload.password)
    raw_token, auth_session = create_session(db, user, settings)
    set_session_cookies(response, raw_token, auth_session, settings)
    return _user_out(user, settings)


@router.post("/auth/login", response_model=UserOut)
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> UserOut:
    _limit(request, "login", 10, 900)
    user = authenticate_credentials(db, str(payload.email), payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    raw_token, auth_session = create_session(db, user, settings)
    set_session_cookies(response, raw_token, auth_session, settings)
    return _user_out(user, settings)


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    response: Response,
    context: AuthContext = Depends(require_csrf),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Response:
    revoke_session(db, context.session)
    clear_session_cookies(response, settings)
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.get("/me", response_model=UserOut)
def me(
    context: AuthContext = Depends(require_auth),
    settings: Settings = Depends(get_settings),
) -> UserOut:
    return _user_out(context.user, settings)


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request,
    context: AuthContext = Depends(require_csrf),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    _limit(request, f"chat:{context.user.id}", 30, 600)
    service = ChatService(
        settings=settings,
        retriever=Retriever(get_embedding_provider()),
        model=get_model_provider(),
    )
    return service.answer(
        db,
        context.user,
        payload.message,
        payload.care_mode,
        payload.conversation_id,
    )


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
def delete_history(
    context: AuthContext = Depends(require_csrf),
    db: Session = Depends(get_db),
) -> Response:
    db.execute(delete(Conversation).where(Conversation.user_id == context.user.id))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/account", status_code=status.HTTP_204_NO_CONTENT)
def delete_account(
    payload: AccountDeleteRequest,
    response: Response,
    context: AuthContext = Depends(require_csrf),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Response:
    if not verify_password(context.user.password_hash, payload.password):
        raise HTTPException(status_code=401, detail="Password confirmation failed")
    db.delete(context.user)
    db.commit()
    clear_session_cookies(response, settings)
    response.status_code = status.HTTP_204_NO_CONTENT
    return response
