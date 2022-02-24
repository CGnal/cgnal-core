"""Definition of base exceptions."""


class NoTableException(BaseException):
    """Exception representing a non existing table."""

    def __init__(self, message: str) -> None:
        """
        Return an instance of the exception.

        :param message: output message
        """
        super(NoTableException, self).__init__(message)
