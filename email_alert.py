import smtplib
from email.mime.text import MIMEText

def send_email_alert(user, maps_link, doctor_email, sender_email, sender_password):
    receiver = doctor_email
    subject = f"🚨 Seizure Alert for {user}"
    body = f"{user} has had a seizure.\n\nLive location: {maps_link}"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver, msg.as_string())
        print(f"✅ Email sent to {receiver}")
    except Exception as e:
        print(f"❌ Email failed: {e}")
