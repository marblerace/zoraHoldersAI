import pytest
from pydantic import ValidationError

from app.config import Settings


def test_token_address_is_normalized() -> None:
    settings = Settings(
        _env_file=None,
        tracked_token_address="0x7777777D57C1C6E472fa379b7B3B6C6Ba3835073",
    )

    assert settings.tracked_token_address == "0x7777777d57c1c6e472fa379b7b3b6c6ba3835073"


def test_invalid_token_address_is_rejected() -> None:
    with pytest.raises(ValidationError, match="20-byte"):
        Settings(_env_file=None, tracked_token_address="not-an-address")
