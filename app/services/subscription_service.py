from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, List
import logging

from app.models import Subscription, BotType, Client
from app.schemas import SubscriptionCreate
from app.utils.date_utils import calculate_expiry_date, is_subscription_active, get_remaining_days

logger = logging.getLogger(__name__)

class SubscriptionService:
    @staticmethod
    def add_subscription_months(db: Session, subscription_data: SubscriptionCreate) -> Subscription:
        """
        Add months to a subscription with the exact logic you specified:
        - First payment: Sets start_date = today, expiry_date = today + months
        - Renewal: Extends expiry_date by months, keeps start_date unchanged
        """
        # Check if subscription already exists
        existing_sub = db.query(Subscription).filter(
            Subscription.client_id == subscription_data.client_id,
            Subscription.bot_type == subscription_data.bot_type
        ).first()

        current_date = datetime.now().date()
        
        if existing_sub:
            # Renewal scenario
            if existing_sub.expiry_date and existing_sub.expiry_date > current_date:
                # Subscription is still active - extend from current expiry date
                new_expiry_date = calculate_expiry_date(
                    existing_sub.expiry_date, 
                    subscription_data.months
                )
                existing_sub.expiry_date = new_expiry_date
            else:
                # Subscription expired or never had dates - treat as new
                existing_sub.start_date = current_date
                existing_sub.expiry_date = calculate_expiry_date(
                    current_date, 
                    subscription_data.months
                )
            
            existing_sub.active = True
            db.commit()
            db.refresh(existing_sub)
            
            logger.info(f"Renewed subscription for client {subscription_data.client_id}, "
                       f"bot {subscription_data.bot_type} for {subscription_data.months} months")
            return existing_sub
        else:
            # First payment scenario
            new_subscription = Subscription(
                client_id=subscription_data.client_id,
                bot_type=subscription_data.bot_type,
                start_date=current_date,
                expiry_date=calculate_expiry_date(current_date, subscription_data.months),
                active=True
            )
            
            db.add(new_subscription)
            db.commit()
            db.refresh(new_subscription)
            
            logger.info(f"Created new subscription for client {subscription_data.client_id}, "
                       f"bot {subscription_data.bot_type} for {subscription_data.months} months")
            return new_subscription

    @staticmethod
    def get_subscription(db: Session, client_id: int, bot_type: str) -> Optional[Subscription]:
        """Get subscription for a specific client and bot type"""
        return db.query(Subscription).filter(
            Subscription.client_id == client_id,
            Subscription.bot_type == bot_type
        ).first()

    @staticmethod
    def get_all_subscriptions(db: Session, client_id: int) -> List[Subscription]:
        """Get all subscriptions for a client"""
        return db.query(Subscription).filter(
            Subscription.client_id == client_id
        ).all()

    @staticmethod
    def deactivate_subscription(db: Session, client_id: int, bot_type: str) -> None:
        """Deactivate a subscription (manual deactivation)"""
        subscription = db.query(Subscription).filter(
            Subscription.client_id == client_id,
            Subscription.bot_type == bot_type
        ).first()
        
        if subscription:
            subscription.active = False
            db.commit()
            logger.info(f"Deactivated subscription for client {client_id}, bot {bot_type}")

    @staticmethod
    def reactivate_subscription(db: Session, client_id: int, bot_type: str) -> None:
        """Reactivate a subscription if it's not expired"""
        subscription = db.query(Subscription).filter(
            Subscription.client_id == client_id,
            Subscription.bot_type == bot_type
        ).first()
        
        if subscription and subscription.expiry_date and subscription.expiry_date > datetime.now().date():
            subscription.active = True
            db.commit()
            logger.info(f"Reactivated subscription for client {client_id}, bot {bot_type}")
        else:
            logger.warning(f"Cannot reactivate expired subscription for client {client_id}, bot {bot_type}")

    @staticmethod
    def check_and_update_expiry(db: Session, client_id: int) -> None:
        """
        Check and update expiry status for all subscriptions of a client
        This should be called daily via a cron job
        """
        subscriptions = db.query(Subscription).filter(
            Subscription.client_id == client_id
        ).all()
        
        current_date = datetime.now().date()
        
        for subscription in subscriptions:
            if subscription.expiry_date and subscription.expiry_date <= current_date:
                subscription.active = False
                logger.info(f"Subscription expired for client {client_id}, bot {subscription.bot_type}")
        
        db.commit()

    @staticmethod
    def check_all_expiries(db: Session) -> None:
        """Check expiry status for all subscriptions in the system (for cron job)"""
        current_date = datetime.now().date()
        expired_subscriptions = db.query(Subscription).filter(
            Subscription.expiry_date <= current_date,
            Subscription.active == True
        ).all()
        
        for subscription in expired_subscriptions:
            subscription.active = False
            logger.info(f"Subscription auto-expired for client {subscription.client_id}, "
                       f"bot {subscription.bot_type}")
        
        db.commit()

    @staticmethod
    def get_subscription_status(db: Session, client_id: int, bot_type: str) -> dict:
        """Get detailed subscription status for display"""
        subscription = db.query(Subscription).filter(
            Subscription.client_id == client_id,
            Subscription.bot_type == bot_type
        ).first()
        
        if not subscription:
            return {
                "status": "INACTIVE",
                "start_date": None,
                "expiry_date": None,
                "remaining_days": 0,
                "active": False
            }
        
        current_date = datetime.now().date()
        active = subscription.active and (
            subscription.expiry_date is None or 
            subscription.expiry_date > current_date
        )
        
        remaining_days = get_remaining_days(subscription.expiry_date) if subscription.expiry_date else 0
        
        if not active:
            status = "EXPIRED" if subscription.expiry_date and subscription.expiry_date <= current_date else "INACTIVE"
        else:
            status = "ACTIVE"
        
        return {
            "status": status,
            "start_date": subscription.start_date,
            "expiry_date": subscription.expiry_date,
            "remaining_days": remaining_days,
            "active": active
        }

    @staticmethod
    def set_custom_subscription(db: Session, client_id: int, bot_type: str, 
                              start_date: datetime, expiry_date: datetime) -> Subscription:
        """Set custom subscription dates (for manual adjustments)"""
        subscription = db.query(Subscription).filter(
            Subscription.client_id == client_id,
            Subscription.bot_type == bot_type
        ).first()
        
        if subscription:
            subscription.start_date = start_date
            subscription.expiry_date = expiry_date
            subscription.active = expiry_date > datetime.now().date() if expiry_date else False
        else:
            subscription = Subscription(
                client_id=client_id,
                bot_type=bot_type,
                start_date=start_date,
                expiry_date=expiry_date,
                active=expiry_date > datetime.now().date() if expiry_date else False
            )
            db.add(subscription)
        
        db.commit()
        db.refresh(subscription)
        return subscription

    @staticmethod
    def get_client_active_subscriptions(db: Session, client_id: int) -> List[Subscription]:
        """Get all active subscriptions for a client"""
        current_date = datetime.now().date()
        return db.query(Subscription).filter(
            Subscription.client_id == client_id,
            Subscription.active == True,
            Subscription.expiry_date > current_date
        ).all()

    @staticmethod
    def get_upcoming_expiries(db: Session, days: int = 7) -> List[Subscription]:
        """Get subscriptions expiring in the next X days (for notifications)"""
        current_date = datetime.now().date()
        target_date = current_date + timedelta(days=days)
        
        return db.query(Subscription).filter(
            Subscription.expiry_date.between(current_date, target_date),
            Subscription.active == True
        ).all()
