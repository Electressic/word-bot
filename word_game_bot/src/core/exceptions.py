"""Custom exceptions for the word game bot"""


class WordGameBotException(Exception):
    """Base exception for word game bot"""
    pass


class DetectionError(WordGameBotException):
    """Exception raised when letter detection fails"""
    pass


class AutomationError(WordGameBotException):
    """Exception raised when automation fails"""
    pass
