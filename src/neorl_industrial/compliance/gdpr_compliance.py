"""GDPR compliance framework for neoRL-industrial-gym."""

import json
import time
import uuid
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import logging

class DataCategory(Enum):
    """Categories of personal data under GDPR."""
    PERSONAL_IDENTIFIABLE = "personal_identifiable"
    SPECIAL_CATEGORY = "special_category"  # Art. 9 GDPR
    CRIMINAL_DATA = "criminal_data"  # Art. 10 GDPR
    PSEUDONYMOUS = "pseudonymous"
    ANONYMOUS = "anonymous"

class ProcessingPurpose(Enum):
    """Lawful purposes for data processing under GDPR."""
    CONSENT = "consent"  # Art. 6(1)(a)
    CONTRACT = "contract"  # Art. 6(1)(b)
    LEGAL_OBLIGATION = "legal_obligation"  # Art. 6(1)(c)
    VITAL_INTERESTS = "vital_interests"  # Art. 6(1)(d)
    PUBLIC_TASK = "public_task"  # Art. 6(1)(e)
    LEGITIMATE_INTERESTS = "legitimate_interests"  # Art. 6(1)(f)

@dataclass
class DataRecord:
    """Record of personal data processing."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_subject_id: str = ""
    data_category: DataCategory = DataCategory.ANONYMOUS
    processing_purpose: ProcessingPurpose = ProcessingPurpose.LEGITIMATE_INTERESTS
    data_source: str = ""
    processing_timestamp: float = field(default_factory=time.time)
    retention_period_days: int = 365
    consent_given: bool = False
    consent_timestamp: Optional[float] = None
    consent_withdrawn: bool = False
    consent_withdrawal_timestamp: Optional[float] = None
    processed_data_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConsentRecord:
    """Record of data subject consent."""
    consent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_subject_id: str = ""
    processing_purposes: List[ProcessingPurpose] = field(default_factory=list)
    consent_timestamp: float = field(default_factory=time.time)
    consent_text: str = ""
    consent_version: str = "1.0"
    is_withdrawn: bool = False
    withdrawal_timestamp: Optional[float] = None
    withdrawal_reason: str = ""

class GDPRDataManager:
    """GDPR-compliant data management system."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize GDPR data manager."""
        self.storage_path = storage_path or Path("gdpr_data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Data records
        self.data_records: Dict[str, DataRecord] = {}
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.data_subject_index: Dict[str, Set[str]] = {}  # subject_id -> record_ids
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Logging
        self.logger = logging.getLogger("gdpr_manager")
        
        # Load existing records
        self._load_records()
        
        # Auto-cleanup thread
        self._cleanup_thread = threading.Thread(target=self._auto_cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def record_data_processing(
        self,
        data_subject_id: str,
        data_category: DataCategory,
        processing_purpose: ProcessingPurpose,
        data_source: str,
        retention_days: int = 365,
        consent_required: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record data processing activity."""
        
        with self._lock:
            # Check consent if required
            if consent_required and not self._has_valid_consent(data_subject_id, processing_purpose):
                raise ValueError(
                    f"No valid consent for data subject {data_subject_id} "
                    f"and purpose {processing_purpose}"
                )
            
            # Create data record
            record = DataRecord(
                data_subject_id=data_subject_id,
                data_category=data_category,
                processing_purpose=processing_purpose,
                data_source=data_source,
                retention_period_days=retention_days,
                consent_given=not consent_required or self._has_valid_consent(data_subject_id, processing_purpose),
                metadata=metadata or {}
            )
            
            # Store record
            self.data_records[record.record_id] = record
            
            # Update subject index
            if data_subject_id not in self.data_subject_index:
                self.data_subject_index[data_subject_id] = set()
            self.data_subject_index[data_subject_id].add(record.record_id)
            
            # Persist changes
            self._save_records()
            
            self.logger.info(f"Recorded data processing: {record.record_id} for subject {data_subject_id}")
            
            return record.record_id
    
    def record_consent(
        self,
        data_subject_id: str,
        processing_purposes: List[ProcessingPurpose],
        consent_text: str,
        consent_version: str = "1.0"
    ) -> str:
        """Record data subject consent."""
        
        with self._lock:
            consent = ConsentRecord(
                data_subject_id=data_subject_id,
                processing_purposes=processing_purposes,
                consent_text=consent_text,
                consent_version=consent_version
            )
            
            self.consent_records[consent.consent_id] = consent
            self._save_records()
            
            self.logger.info(f"Recorded consent: {consent.consent_id} for subject {data_subject_id}")
            
            return consent.consent_id
    
    def withdraw_consent(
        self,
        data_subject_id: str,
        consent_id: Optional[str] = None,
        withdrawal_reason: str = ""
    ) -> bool:
        """Withdraw consent for data processing."""
        
        with self._lock:
            if consent_id:
                # Withdraw specific consent
                if consent_id in self.consent_records:
                    consent = self.consent_records[consent_id]
                    if consent.data_subject_id == data_subject_id:
                        consent.is_withdrawn = True
                        consent.withdrawal_timestamp = time.time()
                        consent.withdrawal_reason = withdrawal_reason
                        
                        self._save_records()
                        
                        self.logger.info(f"Consent withdrawn: {consent_id} for subject {data_subject_id}")
                        return True
            else:
                # Withdraw all consent for subject
                withdrawn_count = 0
                for consent in self.consent_records.values():
                    if consent.data_subject_id == data_subject_id and not consent.is_withdrawn:
                        consent.is_withdrawn = True
                        consent.withdrawal_timestamp = time.time()
                        consent.withdrawal_reason = withdrawal_reason
                        withdrawn_count += 1
                
                if withdrawn_count > 0:
                    self._save_records()
                    self.logger.info(f"All consent withdrawn for subject {data_subject_id}: {withdrawn_count} records")
                    return True
            
            return False
    
    def get_subject_data(self, data_subject_id: str) -> Dict[str, Any]:
        """Get all data for a data subject (Right of Access - Art. 15 GDPR)."""
        
        with self._lock:
            if data_subject_id not in self.data_subject_index:
                return {"data_records": [], "consent_records": []}
            
            # Get data records
            record_ids = self.data_subject_index[data_subject_id]
            data_records = [
                self._serialize_data_record(self.data_records[rid])
                for rid in record_ids
                if rid in self.data_records
            ]
            
            # Get consent records
            consent_records = [
                self._serialize_consent_record(consent)
                for consent in self.consent_records.values()
                if consent.data_subject_id == data_subject_id
            ]
            
            return {
                "data_subject_id": data_subject_id,
                "data_records": data_records,
                "consent_records": consent_records,
                "export_timestamp": time.time(),
                "export_format": "JSON"
            }
    
    def delete_subject_data(self, data_subject_id: str, verify_consent_withdrawal: bool = True) -> bool:
        """Delete all data for a data subject (Right to Erasure - Art. 17 GDPR)."""
        
        with self._lock:
            if data_subject_id not in self.data_subject_index:
                return False
            
            # Check if consent has been withdrawn (if required)
            if verify_consent_withdrawal:
                has_active_consent = any(
                    not consent.is_withdrawn
                    for consent in self.consent_records.values()
                    if consent.data_subject_id == data_subject_id
                )
                
                if has_active_consent:
                    self.logger.warning(f"Cannot delete data for {data_subject_id}: active consent exists")
                    return False
            
            # Delete data records
            record_ids = self.data_subject_index[data_subject_id].copy()
            deleted_count = 0
            
            for record_id in record_ids:
                if record_id in self.data_records:
                    del self.data_records[record_id]
                    deleted_count += 1
            
            # Delete consent records
            consent_ids_to_delete = [
                cid for cid, consent in self.consent_records.items()
                if consent.data_subject_id == data_subject_id
            ]
            
            for consent_id in consent_ids_to_delete:
                del self.consent_records[consent_id]
            
            # Remove from subject index
            del self.data_subject_index[data_subject_id]
            
            # Persist changes
            self._save_records()
            
            self.logger.info(
                f"Deleted all data for subject {data_subject_id}: "
                f"{deleted_count} data records, {len(consent_ids_to_delete)} consent records"
            )
            
            return True
    
    def anonymize_subject_data(self, data_subject_id: str) -> bool:
        """Anonymize data for a data subject."""
        
        with self._lock:
            if data_subject_id not in self.data_subject_index:
                return False
            
            # Generate anonymous ID
            anonymous_id = f"anon_{uuid.uuid4().hex[:8]}"
            
            # Update data records
            record_ids = self.data_subject_index[data_subject_id].copy()
            
            for record_id in record_ids:
                if record_id in self.data_records:
                    record = self.data_records[record_id]
                    record.data_subject_id = anonymous_id
                    record.data_category = DataCategory.ANONYMOUS
                    record.consent_given = False  # No longer applicable
            
            # Update consent records
            for consent in self.consent_records.values():
                if consent.data_subject_id == data_subject_id:
                    consent.data_subject_id = anonymous_id
            
            # Update subject index
            self.data_subject_index[anonymous_id] = self.data_subject_index[data_subject_id]
            del self.data_subject_index[data_subject_id]
            
            # Persist changes
            self._save_records()
            
            self.logger.info(f"Anonymized data for subject {data_subject_id} -> {anonymous_id}")
            
            return True
    
    def _has_valid_consent(self, data_subject_id: str, purpose: ProcessingPurpose) -> bool:
        """Check if data subject has valid consent for purpose."""
        
        for consent in self.consent_records.values():
            if (consent.data_subject_id == data_subject_id and
                not consent.is_withdrawn and
                purpose in consent.processing_purposes):
                return True
        
        return False
    
    def _auto_cleanup_worker(self):
        """Background worker for automatic data cleanup."""
        while True:
            try:
                time.sleep(3600)  # Check every hour
                self._cleanup_expired_data()
            except Exception as e:
                self.logger.error(f"Auto-cleanup error: {e}")
    
    def _cleanup_expired_data(self):
        """Clean up data that has exceeded retention period."""
        
        with self._lock:
            current_time = time.time()
            expired_records = []
            
            for record_id, record in self.data_records.items():
                # Calculate expiration time
                retention_seconds = record.retention_period_days * 24 * 3600
                expiration_time = record.processing_timestamp + retention_seconds
                
                if current_time > expiration_time:
                    expired_records.append(record_id)
            
            # Delete expired records
            for record_id in expired_records:
                record = self.data_records[record_id]
                subject_id = record.data_subject_id
                
                # Remove from data records
                del self.data_records[record_id]
                
                # Update subject index
                if subject_id in self.data_subject_index:
                    self.data_subject_index[subject_id].discard(record_id)
                    if not self.data_subject_index[subject_id]:
                        del self.data_subject_index[subject_id]
            
            if expired_records:
                self._save_records()
                self.logger.info(f"Auto-cleaned {len(expired_records)} expired data records")
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate GDPR compliance report."""
        
        with self._lock:
            current_time = time.time()
            
            # Data processing statistics
            total_records = len(self.data_records)
            records_by_category = {}
            records_by_purpose = {}
            
            for record in self.data_records.values():
                # Category statistics
                category = record.data_category.value
                records_by_category[category] = records_by_category.get(category, 0) + 1
                
                # Purpose statistics
                purpose = record.processing_purpose.value
                records_by_purpose[purpose] = records_by_purpose.get(purpose, 0) + 1
            
            # Consent statistics
            total_consents = len(self.consent_records)
            active_consents = sum(
                1 for consent in self.consent_records.values()
                if not consent.is_withdrawn
            )
            withdrawn_consents = total_consents - active_consents
            
            # Subject statistics
            total_subjects = len(self.data_subject_index)
            
            # Retention compliance
            retention_stats = self._analyze_retention_compliance()
            
            return {
                "report_timestamp": current_time,
                "data_processing": {
                    "total_records": total_records,
                    "records_by_category": records_by_category,
                    "records_by_purpose": records_by_purpose,
                    "total_data_subjects": total_subjects,
                },
                "consent_management": {
                    "total_consents": total_consents,
                    "active_consents": active_consents,
                    "withdrawn_consents": withdrawn_consents,
                    "consent_rate": active_consents / max(total_consents, 1),
                },
                "retention_compliance": retention_stats,
                "gdpr_rights_exercised": {
                    "data_access_requests": 0,  # Would track in practice
                    "erasure_requests": 0,
                    "portability_requests": 0,
                },
            }
    
    def _analyze_retention_compliance(self) -> Dict[str, Any]:
        """Analyze retention period compliance."""
        
        current_time = time.time()
        
        total_records = len(self.data_records)
        expired_records = 0
        expiring_soon = 0  # Within 30 days
        
        for record in self.data_records.values():
            retention_seconds = record.retention_period_days * 24 * 3600
            expiration_time = record.processing_timestamp + retention_seconds
            
            if current_time > expiration_time:
                expired_records += 1
            elif current_time > (expiration_time - 30 * 24 * 3600):
                expiring_soon += 1
        
        return {
            "total_records": total_records,
            "expired_records": expired_records,
            "expiring_soon": expiring_soon,
            "compliance_rate": (total_records - expired_records) / max(total_records, 1),
        }
    
    def _serialize_data_record(self, record: DataRecord) -> Dict[str, Any]:
        """Serialize data record for export."""
        return {
            "record_id": record.record_id,
            "data_category": record.data_category.value,
            "processing_purpose": record.processing_purpose.value,
            "data_source": record.data_source,
            "processing_timestamp": record.processing_timestamp,
            "retention_period_days": record.retention_period_days,
            "consent_given": record.consent_given,
            "metadata": record.metadata,
        }
    
    def _serialize_consent_record(self, consent: ConsentRecord) -> Dict[str, Any]:
        """Serialize consent record for export."""
        return {
            "consent_id": consent.consent_id,
            "processing_purposes": [p.value for p in consent.processing_purposes],
            "consent_timestamp": consent.consent_timestamp,
            "consent_text": consent.consent_text,
            "consent_version": consent.consent_version,
            "is_withdrawn": consent.is_withdrawn,
            "withdrawal_timestamp": consent.withdrawal_timestamp,
            "withdrawal_reason": consent.withdrawal_reason,
        }
    
    def _save_records(self):
        """Persist records to storage."""
        try:
            # Save data records
            data_file = self.storage_path / "data_records.json"
            with open(data_file, "w") as f:
                serialized_data = {
                    rid: self._serialize_data_record(record)
                    for rid, record in self.data_records.items()
                }
                json.dump(serialized_data, f, indent=2)
            
            # Save consent records
            consent_file = self.storage_path / "consent_records.json"
            with open(consent_file, "w") as f:
                serialized_consent = {
                    cid: self._serialize_consent_record(consent)
                    for cid, consent in self.consent_records.items()
                }
                json.dump(serialized_consent, f, indent=2)
            
            # Save subject index
            index_file = self.storage_path / "subject_index.json"
            with open(index_file, "w") as f:
                serialized_index = {
                    subject_id: list(record_ids)
                    for subject_id, record_ids in self.data_subject_index.items()
                }
                json.dump(serialized_index, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save GDPR records: {e}")
    
    def _load_records(self):
        """Load records from storage."""
        try:
            # Load data records
            data_file = self.storage_path / "data_records.json"
            if data_file.exists():
                with open(data_file, "r") as f:
                    data = json.load(f)
                    for rid, record_data in data.items():
                        record = DataRecord(
                            record_id=rid,
                            data_subject_id=record_data["data_subject_id"],
                            data_category=DataCategory(record_data["data_category"]),
                            processing_purpose=ProcessingPurpose(record_data["processing_purpose"]),
                            data_source=record_data["data_source"],
                            processing_timestamp=record_data["processing_timestamp"],
                            retention_period_days=record_data["retention_period_days"],
                            consent_given=record_data["consent_given"],
                            metadata=record_data.get("metadata", {})
                        )
                        self.data_records[rid] = record
            
            # Load consent records
            consent_file = self.storage_path / "consent_records.json"
            if consent_file.exists():
                with open(consent_file, "r") as f:
                    data = json.load(f)
                    for cid, consent_data in data.items():
                        consent = ConsentRecord(
                            consent_id=cid,
                            data_subject_id=consent_data["data_subject_id"],
                            processing_purposes=[ProcessingPurpose(p) for p in consent_data["processing_purposes"]],
                            consent_timestamp=consent_data["consent_timestamp"],
                            consent_text=consent_data["consent_text"],
                            consent_version=consent_data["consent_version"],
                            is_withdrawn=consent_data["is_withdrawn"],
                            withdrawal_timestamp=consent_data.get("withdrawal_timestamp"),
                            withdrawal_reason=consent_data.get("withdrawal_reason", "")
                        )
                        self.consent_records[cid] = consent
            
            # Load subject index
            index_file = self.storage_path / "subject_index.json"
            if index_file.exists():
                with open(index_file, "r") as f:
                    data = json.load(f)
                    for subject_id, record_ids in data.items():
                        self.data_subject_index[subject_id] = set(record_ids)
                        
        except Exception as e:
            self.logger.error(f"Failed to load GDPR records: {e}")

# Global GDPR manager instance
_gdpr_manager = None

def get_gdpr_manager() -> GDPRDataManager:
    """Get global GDPR manager instance."""
    global _gdpr_manager
    if _gdpr_manager is None:
        _gdpr_manager = GDPRDataManager()
    return _gdpr_manager