import os
import logging
import requests
from typing import Tuple, Dict, Any

from django.conf import settings
from django.core.mail import send_mail
from django.db import transaction
from django.utils import timezone
from celery import shared_task

from base.models import AlertDispatchLog

logger = logging.getLogger(__name__)


# ==========================================
# 1. GATEWAY INTEGRATIONS
# ==========================================

class AfricaTalkingSMSClient:
    """Client wrapper for Africa's Talking SMS Gateway (East Africa Primary)."""
    
    def __init__(self):
        self.username = getattr(settings, 'AFRICASTALKING_USERNAME', os.getenv('AFRICASTALKING_USERNAME', 'sandbox'))
        self.api_key = getattr(settings, 'AFRICASTALKING_API_KEY', os.getenv('AFRICASTALKING_API_KEY', ''))
        self.sender_id = getattr(settings, 'AFRICASTALKING_SENDER_ID', os.getenv('AFRICASTALKING_SENDER_ID', 'GeoAfya'))
        
        if self.username == 'sandbox':
            self.url = "https://api.sandbox.africastalking.com/version1/messaging"
        else:
            self.url = "https://api.africastalking.com/version1/messaging"

    def send_sms(self, phone_number: str, message: str) -> Tuple[bool, str]:
        if not self.api_key:
            logger.warning("[SMS Sandbox] Missing AFRICASTALKING_API_KEY. Mocking dispatch to %s", phone_number)
            return True, "MOCK_DISPATCH_SUCCESS_NO_KEY"

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "apiKey": self.api_key
        }
        data = {
            "username": self.username,
            "to": phone_number,
            "message": message,
            "from": self.sender_id
        }

        try:
            response = requests.post(self.url, data=data, headers=headers, timeout=10)
            res_data = response.json()

            if response.status_code in (200, 201):
                recipients = res_data.get('SMSMessageData', {}).get('Recipients', [])
                if recipients and recipients[0].get('status') in ['Success', 'Pending']:
                    msg_id = recipients[0].get('messageId', 'OK')
                    return True, f"MessageID: {msg_id}"
                else:
                    err = recipients[0].get('status', 'Unknown SMS gateway error') if recipients else 'Empty response'
                    return False, f"Gateway error: {err}"
            else:
                return False, f"HTTP {response.status_code}: {response.text}"

        except Exception as e:
            logger.exception("Africa's Talking dispatch exception for %s", phone_number)
            return False, str(e)


class TwilioSMSClient:
    """Fallback SMS Client using Twilio REST API."""

    def __init__(self):
        self.account_sid = getattr(settings, 'TWILIO_ACCOUNT_SID', os.getenv('TWILIO_ACCOUNT_SID', ''))
        self.auth_token = getattr(settings, 'TWILIO_AUTH_TOKEN', os.getenv('TWILIO_AUTH_TOKEN', ''))
        self.from_number = getattr(settings, 'TWILIO_FROM_NUMBER', os.getenv('TWILIO_FROM_NUMBER', ''))

    def send_sms(self, phone_number: str, message: str) -> Tuple[bool, str]:
        if not self.account_sid or not self.auth_token:
            return False, "Twilio credentials missing."

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        auth = (self.account_sid, self.auth_token)
        data = {
            "To": phone_number,
            "From": self.from_number,
            "Body": message
        }

        try:
            response = requests.post(url, data=data, auth=auth, timeout=10)
            if response.status_code in (200, 201):
                return True, f"TwilioSID: {response.json().get('sid')}"
            return False, f"HTTP {response.status_code}: {response.text}"
        except Exception as e:
            return False, str(e)


def send_email_alert(recipient_email: str, subject: str, message: str) -> Tuple[bool, str]:
    """Sends HTML/Plain text epidemiological alert emails via Django SMTP backend."""
    try:
        sent_count = send_mail(
            subject=subject,
            message=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[recipient_email],
            fail_silently=False,
        )
        if sent_count > 0:
            return True, "Email sent successfully"
        return False, "Failed to deliver email"
    except Exception as e:
        logger.exception("Email dispatch exception to %s", recipient_email)
        return False, str(e)


# ==========================================
# 2. CORE DISPATCH SERVICE
# ==========================================

def process_single_dispatch_log(log: AlertDispatchLog) -> Tuple[bool, str]:
    """
    Executes single notification dispatch based on log channel configuration.
    Channel options: 'SMS', 'EMAIL', or 'BOTH'.
    """
    successes = []
    errors = []
    
    # 1. Handle SMS Dispatch
    if log.channel in ['SMS', 'BOTH']:
        if not log.recipient_phone:
            errors.append("Missing recipient phone number for SMS channel.")
        else:
            # Primary SMS provider (Africa's Talking), fallback to Twilio
            at_client = AfricaTalkingSMSClient()
            ok, response_msg = at_client.send_sms(log.recipient_phone, log.alert_message)
            
            if not ok:
                logger.warning("Africa's Talking failed for log %s. Trying Twilio fallback...", log.id)
                twilio_client = TwilioSMSClient()
                ok, response_msg = twilio_client.send_sms(log.recipient_phone, log.alert_message)

            if ok:
                successes.append(f"SMS: {response_msg}")
            else:
                errors.append(f"SMS: {response_msg}")

    # 2. Handle Email Dispatch
    if log.channel in ['EMAIL', 'BOTH']:
        if not log.recipient_email:
            errors.append("Missing recipient email address for EMAIL channel.")
        else:
            subject = f"🚨 GeoAfya Epidemiological Risk Alert [{log.risk_level}] - Cell #{log.spatial_cell.cell_id}"
            ok, response_msg = send_email_alert(log.recipient_email, subject, log.alert_message)
            
            if ok:
                successes.append(f"EMAIL: {response_msg}")
            else:
                errors.append(f"EMAIL: {response_msg}")

    # Consolidate status
    if errors and not successes:
        return False, " | ".join(errors)
    elif errors and successes:
        return True, f"PARTIAL: Success [{'; '.join(successes)}] | Errors [{'; '.join(errors)}]"
    else:
        return True, " | ".join(successes)


# ==========================================
# 3. CELERY ALERT DISPATCH TASK
# ==========================================

@shared_task(
    bind=True,
    name="base.alerts.dispatch_pending_alerts",
    max_retries=3,
    default_retry_delay=60, # 1 minute retry interval
    rate_limit="100/m"       # Prevent API rate limiting on SMS gateways
)
def dispatch_pending_alerts(self, batch_size: int = 100) -> Dict[str, Any]:
    """
    Celery Task: Queries pending AlertDispatchLog entries, acquires row-level locks,
    dispatches outbound notifications, and updates database execution status atomically.
    """
    logger.info("Starting dispatch_pending_alerts Celery worker run...")
    
    # Fetch pending dispatch records using FOR UPDATE SKIP LOCKED to prevent race conditions
    with transaction.atomic():
        pending_logs = list(
            AlertDispatchLog.objects.select_for_update(skip_locked=True)
            .filter(dispatch_status='PENDING')
            .select_related('spatial_cell', 'risk_assessment')[:batch_size]
        )

    if not pending_logs:
        logger.info("No pending alert dispatches found.")
        return {"processed": 0, "sent": 0, "failed": 0}

    sent_count = 0
    failed_count = 0

    for log in pending_logs:
        try:
            success, detail = process_single_dispatch_log(log)

            with transaction.atomic():
                if success:
                    log.dispatch_status = 'SENT'
                    log.error_message = detail
                    log.sent_at = timezone.now()
                    sent_count += 1
                else:
                    log.retry_count += 1
                    log.error_message = detail
                    if log.retry_count >= self.max_retries:
                        log.dispatch_status = 'FAILED'
                    else:
                        log.dispatch_status = 'PENDING'  # Keep pending for next retry task
                    failed_count += 1

                log.save(update_fields=['dispatch_status', 'error_message', 'sent_at', 'retry_count'])

        except Exception as e:
            logger.exception("Unhandled error processing alert log %s", log.id)
            with transaction.atomic():
                log.retry_count += 1
                log.error_message = f"Unhandled Exception: {str(e)}"
                log.dispatch_status = 'FAILED' if log.retry_count >= self.max_retries else 'PENDING'
                log.save(update_fields=['dispatch_status', 'error_message', 'retry_count'])
            failed_count += 1

    summary = {
        "processed": len(pending_logs),
        "sent": sent_count,
        "failed": failed_count
    }
    logger.info("Alert dispatch run complete: %s", summary)
    return summary