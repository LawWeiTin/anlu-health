import hashlib
import secrets
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from cryptography.fernet import Fernet

_password_hasher = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)


def normalize_email(email: str) -> str:
    return email.strip().casefold()


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password_hash: str, candidate: str) -> bool:
    try:
        return _password_hasher.verify(password_hash, candidate)
    except (VerifyMismatchError, InvalidHashError):
        return False


def random_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def short_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Cipher:
    key: str

    def encrypt(self, value: str) -> bytes:
        return Fernet(self.key.encode("ascii")).encrypt(value.encode("utf-8"))

    def decrypt(self, value: bytes) -> str:
        return Fernet(self.key.encode("ascii")).decrypt(value).decode("utf-8")
