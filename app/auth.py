import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import Depends, HTTPException, Request, Response, status
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings
from app.database import get_db
from app.models import AuthSession, User, utcnow
from app.security import hash_password, hash_token, normalize_email, random_token, verify_password

SESSION_COOKIE = "anlu_session"
CSRF_COOKIE = "anlu_csrf"


@dataclass(frozen=True)
class AuthContext:
    user: User
    session: AuthSession


def create_user(db: Session, email: str, password: str) -> User:
    user = User(
        email=normalize_email(email),
        password_hash=hash_password(password),
        terms_accepted_at=utcnow(),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409, detail="An account with this email already exists"
        ) from exc
    db.refresh(user)
    return user


def authenticate_credentials(db: Session, email: str, password: str) -> User | None:
    user = db.scalar(select(User).where(User.email == normalize_email(email)))
    if not user or user.is_disabled or not verify_password(user.password_hash, password):
        return None
    return user


def create_session(db: Session, user: User, settings: Settings) -> tuple[str, AuthSession]:
    raw_token = random_token()
    auth_session = AuthSession(
        token_hash=hash_token(raw_token),
        user_id=user.id,
        csrf_token=random_token(),
        expires_at=utcnow() + timedelta(days=settings.session_days),
    )
    db.add(auth_session)
    db.commit()
    db.refresh(auth_session)
    return raw_token, auth_session


def set_session_cookies(
    response: Response, raw_token: str, auth_session: AuthSession, settings: Settings
) -> None:
    max_age = settings.session_days * 24 * 60 * 60
    response.set_cookie(
        SESSION_COOKIE,
        raw_token,
        max_age=max_age,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        path="/",
    )
    response.set_cookie(
        CSRF_COOKIE,
        auth_session.csrf_token,
        max_age=max_age,
        httponly=False,
        secure=settings.cookie_secure,
        samesite="strict",
        path="/",
    )


def clear_session_cookies(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        SESSION_COOKIE, path="/", secure=settings.cookie_secure, samesite="strict"
    )
    response.delete_cookie(CSRF_COOKIE, path="/", secure=settings.cookie_secure, samesite="strict")


def _is_expired(value: datetime) -> bool:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value <= utcnow()


def require_auth(
    request: Request,
    db: Session = Depends(get_db),
) -> AuthContext:
    raw_token = request.cookies.get(SESSION_COOKIE)
    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )
    auth_session = db.get(AuthSession, hash_token(raw_token))
    if not auth_session or _is_expired(auth_session.expires_at):
        if auth_session:
            db.delete(auth_session)
            db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")
    user = db.get(User, auth_session.user_id)
    if not user or user.is_disabled:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )
    return AuthContext(user=user, session=auth_session)


def require_csrf(request: Request, context: AuthContext = Depends(require_auth)) -> AuthContext:
    supplied = request.headers.get("X-CSRF-Token", "")
    cookie = request.cookies.get(CSRF_COOKIE, "")
    if not supplied or not cookie:
        raise HTTPException(status_code=403, detail="CSRF validation failed")
    if not secrets.compare_digest(supplied, cookie) or not secrets.compare_digest(
        supplied, context.session.csrf_token
    ):
        raise HTTPException(status_code=403, detail="CSRF validation failed")
    return context


def revoke_session(db: Session, auth_session: AuthSession) -> None:
    db.delete(auth_session)
    db.commit()


def revoke_all_sessions(db: Session, user_id: str) -> None:
    db.execute(delete(AuthSession).where(AuthSession.user_id == user_id))
    db.commit()
