"""Basic functionalities for interacting with email servers."""

import smtplib
from typing import List, Optional, Union
from os.path import basename

from cgnal.core.logging.defaults import WithLogging

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication


class EmailSender(WithLogging):
    """Email Sender Utility Class."""

    def __init__(
        self,
        email_address: str,
        username: str,
        password: str,
        smtp_address: str,
        auth_protocol: str = "None",
        port: Optional[int] = None,
    ) -> None:
        """
        Return the instance of the class that acts as client to interact with an email server.

        :param email_address: Sender email address
        :param username: Username for authentication
        :param password: Password for authentication
        :param smtp_address: SMTP server address
        :param auth_protocol: Authentication protocol to use
        :param port: Port of SMTP server
        """
        self.email_address = email_address
        self.username = username
        self.password = password
        self.smtp_address = smtp_address
        self.auth_protocol = auth_protocol
        self.port = port

    def send_mail(
        self,
        text: str,
        subject: str,
        destination: str,
        attachments: Optional[List[str]] = None,
    ) -> None:
        """
        Send email.

        :param text: The text of the email
        :param subject: The subject of the email
        :param destination: The destination email address
        :param attachments: List with the files to send as attachments to the email
        :return: None
        """
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = self.email_address
        msg["To"] = destination
        msg.attach(MIMEText(text))
        if attachments is not None:
            for f in attachments:
                with open(f, "rb") as fil:
                    part = MIMEApplication(fil.read(), Name=basename(f))
                # After the file is closed
                part["Content-Disposition"] = 'attachment; filename="%s"' % basename(f)
                msg.attach(part)

        try:
            if self.auth_protocol == "SSL":
                port = 465 if self.port is None else self.port
                server: Union[smtplib.SMTP_SSL, smtplib.SMTP] = smtplib.SMTP_SSL(
                    self.smtp_address, port=port
                )
            elif self.auth_protocol == "TLS":
                port = 587 if self.port is None else self.port
                server = smtplib.SMTP(self.smtp_address, port=port)
                server.starttls()
            elif self.auth_protocol == "None":
                port = 25 if self.port is None else self.port
                server = smtplib.SMTP(self.smtp_address, port=port)
            else:
                raise Exception(f"{self.auth_protocol} not implemented")
            server.ehlo()
            server.login(self.username, self.password)
            server.sendmail(self.email_address, destination, msg.as_string())
            server.close()
        except Exception as e:
            raise e
