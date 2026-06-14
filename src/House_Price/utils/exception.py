import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Create a detailed error message with file name and line number.
    """
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return str(error)

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = (
        f"Error occurred in file: [{file_name}] "
        f"at line number: [{line_number}] "
        f"with message: [{str(error)}]"
    )

    return error_message


class CustomException(Exception):
    """
    Custom exception class for the House Price project.
    """

    def __init__(self, error: Exception, error_detail: sys):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self) -> str:
        return self.error_message