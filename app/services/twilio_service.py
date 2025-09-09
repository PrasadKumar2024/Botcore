
# app/services/twilio_service.py - Twilio integration service

import os
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import logging
from typing import Optional, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

class TwilioService:
    def __init__(self):
        # Initialize Twilio client with environment variables
        self.client = Client(
            os.getenv('TWILIO_ACCOUNT_SID'),
            os.getenv('TWILIO_AUTH_TOKEN')
        )
    
    def buy_phone_number(self, country_code: str) -> Optional[Dict[str, Any]]:
        """
        Buy a phone number for the specified country
        Returns: { "phone_number": "+1234567890", "sid": "PN123" }
        """
        try:
            # Map country names to Twilio region codes
            country_mapping = {
                "united states": "US",
                "india": "IN",
                "united kingdom": "GB",
                "australia": "AU",
                "canada": "CA",
                "germany": "DE",
                "france": "FR"
            }
            
            region = country_mapping.get(country_code.lower(), "US")
            
            # Search for available numbers
            available_numbers = self.client.available_phone_numbers(region).local.list(limit=1)
            
            if not available_numbers:
                logger.error(f"No numbers available for country: {country_code}")
                return None
            
            # Purchase the first available number
            number_to_buy = available_numbers[0]
            purchased_number = self.client.incoming_phone_numbers.create(
                phone_number=number_to_buy.phone_number
            )
            
            logger.info(f"Purchased number: {purchased_number.phone_number}")
            
            return {
                "phone_number": purchased_number.phone_number,
                "sid": purchased_number.sid,
                "country": country_code
            }
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error buying number: {e}")
            return None
    
    def update_whatsapp_profile(self, phone_number_sid: str, business_name: str, 
                              logo_url: Optional[str] = None) -> bool:
        """
        Update WhatsApp business profile for a number
        """
        try:
            update_params = {
                "friendly_name": business_name
            }
            
            if logo_url:
                update_params["profile_picture_url"] = logo_url
            
            # Update WhatsApp business profile
            self_client.messaging.v1.services(
                os.getenv('TWILIO_WHATSAPP_SERVICE_SID')
            ).phone_numbers(phone_number_sid).update(**update_params)
            
            logger.info(f"Updated WhatsApp profile for SID: {phone_number_sid}")
            return True
            
        except TwilioRestException as e:
            logger.error(f"Twilio profile update error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating profile: {e}")
            return False
    
    def release_phone_number(self, phone_number_sid: str) -> bool:
        """
        Release a phone number
        """
        try:
            self.client.incoming_phone_numbers(phone_number_sid).delete()
            logger.info(f"Released number SID: {phone_number_sid}")
            return True
        except TwilioRestException as e:
            logger.error(f"Twilio release error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error releasing number: {e}")
            return False

# Create global instance
twilio_service = TwilioService()
