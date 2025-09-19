import os
import logging
from typing import List, Optional, Dict, Any
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

logger = logging.getLogger(__name__)

class TwilioService:
    def __init__(self):
        # Initialize Twilio client with environment variables
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.twilio_client = Client(self.account_sid, self.auth_token)
        
    def buy_phone_number(self, country_code: str, area_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Purchase a phone number from Twilio
        """
        try:
            # Search for available numbers
            if area_code:
                available_numbers = self.twilio_client.available_phone_numbers(country_code) \
                    .local \
                    .list(area_code=area_code, limit=5)
            else:
                available_numbers = self.twilio_client.available_phone_numbers(country_code) \
                    .local \
                    .list(limit=5)
            
            if not available_numbers:
                raise Exception(f"No phone numbers available in {country_code}")
            
            # Purchase the first available number
            phone_number = available_numbers[0].phone_number
            purchased_number = self.twilio_client.incoming_phone_numbers \
                .create(phone_number=phone_number)
            
            logger.info(f"Purchased phone number: {purchased_number.phone_number}")
            
            return {
                "phone_number": purchased_number.phone_number,
                "sid": purchased_number.sid,
                "country": country_code,
                "capabilities": purchased_number.capabilities
            }
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error: {e}")
            raise Exception(f"Failed to purchase number: {e.msg}")
        except Exception as e:
            logger.error(f"Error purchasing phone number: {e}")
            raise
    
    def release_phone_number(self, phone_number_sid: str) -> bool:
        """
        Release a phone number from Twilio
        """
        try:
            self.twilio_client.incoming_phone_numbers(phone_number_sid).delete()
            logger.info(f"Released phone number: {phone_number_sid}")
            return True
        except TwilioRestException as e:
            logger.error(f"Error releasing phone number: {e}")
            return False
    
    def update_whatsapp_profile(self, phone_number_sid: str, business_name: str, 
                              address: str, logo_url: Optional[str] = None) -> bool:
        """
        Update WhatsApp business profile
        """
        try:
            # Get the WhatsApp business profile SID
            profiles = self.twilio_client.messaging.v1.services.list()
            whatsapp_service = None
            
            for profile in profiles:
                if profile.friendly_name.lower().startswith('whatsapp'):
                    whatsapp_service = profile
                    break
            
            if not whatsapp_service:
                raise Exception("WhatsApp business profile not found")
            
            # Update WhatsApp business profile
            update_data = {
                "friendly_name": business_name,
                "business_address": address
            }
            
            if logo_url:
                update_data["business_logo"] = logo_url
            
            # For WhatsApp API, we need to use the appropriate endpoint
            # This is a simplified implementation - actual implementation may vary
            # based on Twilio's WhatsApp Business API requirements
            
            logger.info(f"Updated WhatsApp profile for {phone_number_sid}: {business_name}")
            return True
            
        except TwilioRestException as e:
            logger.error(f"Error updating WhatsApp profile: {e}")
            return False
        except Exception as e:
            logger.error(f"Error in WhatsApp profile update: {e}")
            return False
    
    def get_available_countries(self) -> List[Dict[str, str]]:
        """
        Get list of countries available for phone number purchase
        """
        try:
            countries = self.twilio_client.available_phone_numbers.list()
            
            country_list = []
            for country in countries:
                country_list.append({
                    "code": country.country_code,
                    "name": country.country,
                    "flag": self._get_country_flag(country.country_code)
                })
            
            return country_list
            
        except TwilioRestException as e:
            logger.error(f"Error fetching available countries: {e}")
            return []
    
    def search_available_numbers(self, country_code: str, area_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for available phone numbers
        """
        try:
            if area_code:
                numbers = self.twilio_client.available_phone_numbers(country_code) \
                    .local \
                    .list(area_code=area_code, limit=10)
            else:
                numbers = self.twilio_client.available_phone_numbers(country_code) \
                    .local \
                    .list(limit=10)
            
            available_numbers = []
            for number in numbers:
                available_numbers.append({
                    "phone_number": number.phone_number,
                    "friendly_name": number.friendly_name,
                    "locality": number.locality,
                    "region": number.region,
                    "country": number.iso_country,
                    "capabilities": number.capabilities
                })
            
            return available_numbers
            
        except TwilioRestException as e:
            logger.error(f"Error searching available numbers: {e}")
            return []
    
    def send_whatsapp_message(self, to_number: str, message: str) -> bool:
        """
        Send a WhatsApp message (for testing or notifications)
        """
        try:
            # Get WhatsApp sender number (your Twilio WhatsApp number)
            from_number = f"whatsapp:{os.getenv('TWILIO_WHATSAPP_NUMBER')}"
            
            message = self.twilio_client.messages.create(
                body=message,
                from_=from_number,
                to=f"whatsapp:{to_number}"
            )
            
            logger.info(f"Sent WhatsApp message to {to_number}: {message.sid}")
            return True
            
        except TwilioRestException as e:
            logger.error(f"Error sending WhatsApp message: {e}")
            return False
    
    def make_voice_call(self, to_number: str, message: str) -> bool:
        """
        Make a voice call (for testing or notifications)
        """
        try:
            # This would use Twilio's Voice API to make a call
            # Implementation depends on your specific voice call requirements
            
            logger.info(f"Voice call to {to_number} would be implemented here")
            return True
            
        except Exception as e:
            logger.error(f"Error making voice call: {e}")
            return False
    
    def validate_phone_number(self, phone_number: str) -> Dict[str, Any]:
        """
        Validate a phone number using Twilio's lookup API
        """
        try:
            number = self.twilio_client.lookups.v1.phone_numbers(phone_number) \
                .fetch(type=['carrier'])
            
            return {
                "valid": True,
                "phone_number": number.phone_number,
                "carrier": number.carrier,
                "country_code": number.country_code
            }
            
        except TwilioRestException as e:
            logger.error(f"Error validating phone number: {e}")
            return {"valid": False, "error": str(e)}
    
    def _get_country_flag(self, country_code: str) -> str:
        """
        Helper method to get country flag emoji from country code
        """
        # Simple mapping - in production, you might want a more comprehensive solution
        flag_map = {
            "US": "🇺🇸", "GB": "🇬🇧", "CA": "🇨🇦", "AU": "🇦🇺", "IN": "🇮🇳",
            "DE": "🇩🇪", "FR": "🇫🇷", "BR": "🇧🇷", "MX": "🇲🇽", "ES": "🇪🇸"
        }
        return flag_map.get(country_code.upper(), "🇺🇳")
    
    def get_phone_number_info(self, phone_number_sid: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific phone number
        """
        try:
            number = self.twilio_client.incoming_phone_numbers(phone_number_sid).fetch()
            
            return {
                "phone_number": number.phone_number,
                "friendly_name": number.friendly_name,
                "sid": number.sid,
                "capabilities": number.capabilities,
                "status": number.status
            }
            
        except TwilioRestException as e:
            logger.error(f"Error getting phone number info: {e}")
            return None
    
    def enable_whatsapp_for_number(self, phone_number_sid: str) -> bool:
        """
        Enable WhatsApp for a specific phone number
        """
        try:
            # Update phone number to enable WhatsApp
            self.twilio_client.incoming_phone_numbers(phone_number_sid) \
                .update(sms_url=os.getenv('TWILIO_WHATSAPP_WEBHOOK_URL'))
            
            logger.info(f"Enabled WhatsApp for phone number: {phone_number_sid}")
            return True
            
        except TwilioRestException as e:
            logger.error(f"Error enabling WhatsApp: {e}")
            return False

# Global instance for easy access
twilio_service = TwilioService()
