from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage  # Make sure this import is present
from datetime import datetime
import smtplib
import cv2
import os


def send_email(subj, msgs, image_source=None):
    # Email configuration
    sender_email = "vivektemp57@gmail.com"
    sender_password = "ezhw xfbz yemd dcyo"
    recipient_email = 'shreeveeraganapati15@gmail.com'

    # Create message container
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subj

    # Email body
    body = msgs
    msg.attach(MIMEText(body, 'plain'))

    # Attach image if provided
    if image_source is not None:
        try:
            # Generate a unique filename
            filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

            # Check if image_source is a path or a frame
            if isinstance(image_source, str):
                # If it's a string (path), check if file exists
                if not os.path.exists(image_source):
                    print(f"Image file not found: {image_source}")
                    return

                # Read image from path
                with open(image_source, 'rb') as file:
                    image_data = file.read()
                    image = MIMEImage(image_data, name=os.path.basename(image_source))
            else:
                # Assume it's a OpenCV frame
                _, buffer = cv2.imencode('.jpg', image_source)
                image = MIMEImage(buffer.tobytes(), name=filename)

            # Attach the image
            msg.attach(image)
        except Exception as e:
            print(f"Error attaching image: {e}")
            # Print the full traceback to get more details
            import traceback
            traceback.print_exc()

    try:
        # Establish a secure session with Gmail's outgoing SMTP server using your gmail account
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Secure the connection
        server.login(sender_email, sender_password)

        # Send email
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)

        print("Email sent successfully!")

    except Exception as e:
        print(f"Error sending email: {e}")
        # Print the full traceback to get more details
        import traceback
        traceback.print_exc()

    finally:
        # Close the server connection
        server.quit()


# Example usage
if __name__ == "__main__":
    # Uncomment and modify as needed
    send_email("Test Subject", "Test Message", r"images/3.png")
    send_email("Test Subject", "Test Message")