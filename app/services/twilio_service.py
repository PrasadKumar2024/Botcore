import os
from twilio.rest import Client as TwilioClient
from twilio.base.exceptions import TwilioRestException
import logging
from typing import Optional, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

class TwilioService:
    def __init__(self):
        # Initialize Twilio client with environment variables
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        
        if not account_sid or not auth_token:
            raise ValueError("Twilio credentials not found in environment variables")
        
        self.client = TwilioClient(account_sid, auth_token)
    
    def buy_phone_number(self, country: str) -> Optional[Dict[str, Any]]:
        """
        Buy a phone number for the specified country
        Returns: { "phone_number": "+1234567890", "sid": "PN123", "country": "US" }
        """
        try:
            # Map country names to ISO codes
            country_mapping = {
                "united states": "US",
                "india": "IN",
                "united kingdom": "GB",
                "australia": "AU",
                "canada": "CA",
                "germany": "DE",
                "france": "FR"
            }
            
            # Get ISO code or use the provided value if already in correct format
            iso_code = country_mapping.get(country.lower(), country.upper())
            
            # Search for available numbers
            available_numbers = self.client.available_phone_numbers(iso_code).local.list(limit=5)
            
            if not available_numbers:
                logger.error(f"No phone numbers available for country: {country}")
                return None
            
            # Purchase the first available number
            number_to_buy = available_numbers[0]
            purchased_number = self.client.incoming_phone_numbers.create(
                phone_number=number_to_buy.phone_number
            )
            
            logger.info(f"Purchased Twilio number: {purchased_number.phone_number}")
            
            return {
                "phone_number": purchased_number.phone_number,
                "sid": purchased_number.sid,
                "country": country
            }
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error while buying number: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error buying phone number: {e}")
            return None
    
    def configure_phone_number(self, phone_number_sid: str, voice_url: str, sms_url: str) -> bool:
        """
        Configure a phone number for voice and SMS/WhatsApp
        """
        try:
            # Update phone number configuration
            self.client.incoming_phone_numbers(phone_number_sid).update(
                voice_url=voice_url,
                voice_method='POST',
                sms_url=sms_url,
                sms_method='POST'
            )
            
            logger.info(f"Configured phone number {phone_number_sid} for voice and SMS")
            return True
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error configuring number: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error configuring phone number: {e}")
            return False
    
    def update_whatsapp_profile(self, phone_number_sid: str, business_name: str, 
                              address: Optional[str] = None, 
                              logo_url: Optional[str] = None) -> bool:
        """
        Update WhatsApp business profile for a number
        """
        try:
            # Get WhatsApp service SID from environment
            whatsapp_service_sid = os.getenv('TWILIO_WHATSAPP_SERVICE_SID')
            
            if not whatsapp_service_sid:
                logger.error("WhatsApp Service SID not configured")
                return False
            
            # Update the business profile
            profile = self.client.messaging.v1.services(whatsapp_service_sid) \
                .phone_numbers(phone_number_sid) \
                .update(
                    friendly_name=business_name
                )
            
            # Note: Full business profile update requires additional Twilio permissions
            # and might need to be done manually in Twilio console for some fields
            
            logger.info(f"Updated WhatsApp profile for number {phone_number_sid}")
            return True
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error updating WhatsApp profile: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating WhatsApp profile: {e}")
            return False
    
    def release_phone_number(self, phone_number_sid: str) -> bool:
        """
        Release a phone number
        """
        try:
            self.client.incoming_phone_numbers(phone_number_sid).delete()
            logger.info(f"Released phone number SID: {phone_number_sid}")
            return True
        except TwilioRestException as e:
            logger.error(f"Twilio API error releasing number: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error releasing phone number: {e}")
            return False
    
    def send_whatsapp_message(self, to_number: str, from_number: str, message: str) -> Optional[str]:
        """
        Send a WhatsApp message
        """
        try:
            message = self.client.messages.create(
                body=message,
                from_=f"whatsapp:{from_number}",
                to=f"whatsapp:{to_number}"
            )
            
            logger.info(f"Sent WhatsApp message SID: {message.sid}")
            return message.sid
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error sending WhatsApp message: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error sending WhatsApp message: {e}")
            return None

# Create global instance
twilio_service = TwilioService()
