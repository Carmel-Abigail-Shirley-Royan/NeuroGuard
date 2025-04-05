import smtplib
from email.mime.text import MIMEText
def send_email_alert(user, maps_link, doctor_email, sender_email, sender_password):
    sender = sender_email
    receiver = doctor_email
    subject = f"🚨 Seizure Alert for {user}"
    body = f"{user} has had a seizure.\n\nLive location: {maps_link}"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender, sender_password)  # <-- Must be App Password
            server.sendmail(sender, receiver, msg.as_string())
        print(f"✅ Email sent to {receiver}")
    except Exception as e:
        print(f"❌ Email failed: {e}")


