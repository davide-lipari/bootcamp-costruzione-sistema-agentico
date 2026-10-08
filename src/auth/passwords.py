# src/auth/passwords.py — hash lento e con salt: argon2
from pwdlib import PasswordHash

_hasher = PasswordHash.recommended()


def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _hasher.verify(plain, hashed)
