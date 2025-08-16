"""Internationalization and localization support for global deployment."""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum


class SupportedLocale(Enum):
    """Supported locales for internationalization."""
    EN_US = "en_US"  # English (United States)
    EN_GB = "en_GB"  # English (United Kingdom)
    ES_ES = "es_ES"  # Spanish (Spain)
    ES_MX = "es_MX"  # Spanish (Mexico)
    FR_FR = "fr_FR"  # French (France)
    DE_DE = "de_DE"  # German (Germany)
    JA_JP = "ja_JP"  # Japanese (Japan)
    ZH_CN = "zh_CN"  # Chinese (Simplified)
    ZH_TW = "zh_TW"  # Chinese (Traditional)
    KO_KR = "ko_KR"  # Korean (South Korea)
    PT_BR = "pt_BR"  # Portuguese (Brazil)
    IT_IT = "it_IT"  # Italian (Italy)
    RU_RU = "ru_RU"  # Russian (Russia)
    AR_SA = "ar_SA"  # Arabic (Saudi Arabia)
    HI_IN = "hi_IN"  # Hindi (India)


class LocalizationManager:
    """Manager for internationalization and localization."""
    
    def __init__(self, default_locale: SupportedLocale = SupportedLocale.EN_US):
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Translation storage
        self.translations: Dict[SupportedLocale, Dict[str, str]] = {}
        
        # Load default translations
        self._load_default_translations()
        
        self.logger.info(f"Localization manager initialized with default locale: {default_locale.value}")
        
    def _load_default_translations(self):
        """Load default translation strings for all supported locales."""
        
        # English (US) - Base language
        self.translations[SupportedLocale.EN_US] = {
            # General System Messages
            "system.startup": "System starting up...",
            "system.shutdown": "System shutting down...",
            "system.ready": "System ready",
            "system.error": "System error occurred",
            "system.warning": "System warning",
            
            # Authentication & Access
            "auth.login": "Login",
            "auth.logout": "Logout",
            "auth.unauthorized": "Unauthorized access",
            "auth.invalid_credentials": "Invalid credentials",
            "auth.session_expired": "Session expired",
            
            # Safety & Security
            "safety.violation": "Safety violation detected",
            "safety.critical": "Critical safety alert",
            "safety.emergency_stop": "Emergency stop activated",
            "security.threat_detected": "Security threat detected",
            "security.access_denied": "Access denied",
            
            # Industrial Operations
            "industrial.process_started": "Industrial process started",
            "industrial.process_stopped": "Industrial process stopped",
            "industrial.parameter_out_of_range": "Parameter out of safe range",
            "industrial.maintenance_required": "Maintenance required",
            "industrial.calibration_needed": "Calibration needed",
            
            # Data & Analytics
            "data.loading": "Loading data...",
            "data.processing": "Processing data...",
            "data.analysis_complete": "Analysis complete",
            "data.export_successful": "Data export successful",
            "data.backup_created": "Backup created",
            
            # Training & Models
            "ml.training_started": "Model training started",
            "ml.training_complete": "Model training complete",
            "ml.model_loaded": "Model loaded successfully",
            "ml.prediction_failed": "Prediction failed",
            "ml.model_updated": "Model updated",
            
            # Quality Gates
            "quality.gate_passed": "Quality gate passed",
            "quality.gate_failed": "Quality gate failed", 
            "quality.threshold_exceeded": "Quality threshold exceeded",
            "quality.validation_successful": "Validation successful",
            
            # Compliance & Standards
            "compliance.check_passed": "Compliance check passed",
            "compliance.violation": "Compliance violation",
            "compliance.audit_required": "Audit required",
            "compliance.certification_valid": "Certification valid",
            
            # Error Messages
            "error.connection_failed": "Connection failed",
            "error.timeout": "Operation timed out",
            "error.invalid_input": "Invalid input",
            "error.insufficient_permissions": "Insufficient permissions",
            "error.resource_unavailable": "Resource unavailable",
            
            # Status Messages
            "status.online": "Online",
            "status.offline": "Offline",
            "status.maintenance": "Under maintenance",
            "status.degraded": "Degraded performance",
            "status.operational": "Operational"
        }
        
        # Spanish (Spain)
        self.translations[SupportedLocale.ES_ES] = {
            "system.startup": "Iniciando sistema...",
            "system.shutdown": "Apagando sistema...",
            "system.ready": "Sistema listo",
            "system.error": "Error del sistema",
            "system.warning": "Advertencia del sistema",
            
            "auth.login": "Iniciar sesión",
            "auth.logout": "Cerrar sesión",
            "auth.unauthorized": "Acceso no autorizado",
            "auth.invalid_credentials": "Credenciales inválidas",
            "auth.session_expired": "Sesión expirada",
            
            "safety.violation": "Violación de seguridad detectada",
            "safety.critical": "Alerta crítica de seguridad",
            "safety.emergency_stop": "Parada de emergencia activada",
            "security.threat_detected": "Amenaza de seguridad detectada",
            "security.access_denied": "Acceso denegado",
            
            "industrial.process_started": "Proceso industrial iniciado",
            "industrial.process_stopped": "Proceso industrial detenido",
            "industrial.parameter_out_of_range": "Parámetro fuera del rango seguro",
            "industrial.maintenance_required": "Mantenimiento requerido",
            "industrial.calibration_needed": "Calibración necesaria",
            
            "status.online": "En línea",
            "status.offline": "Fuera de línea",
            "status.maintenance": "En mantenimiento",
            "status.degraded": "Rendimiento degradado",
            "status.operational": "Operacional"
        }
        
        # French (France)
        self.translations[SupportedLocale.FR_FR] = {
            "system.startup": "Démarrage du système...",
            "system.shutdown": "Arrêt du système...",
            "system.ready": "Système prêt",
            "system.error": "Erreur système",
            "system.warning": "Avertissement système",
            
            "auth.login": "Connexion",
            "auth.logout": "Déconnexion",
            "auth.unauthorized": "Accès non autorisé",
            "auth.invalid_credentials": "Identifiants invalides",
            "auth.session_expired": "Session expirée",
            
            "safety.violation": "Violation de sécurité détectée",
            "safety.critical": "Alerte de sécurité critique",
            "safety.emergency_stop": "Arrêt d'urgence activé",
            "security.threat_detected": "Menace de sécurité détectée",
            "security.access_denied": "Accès refusé",
            
            "industrial.process_started": "Processus industriel démarré",
            "industrial.process_stopped": "Processus industriel arrêté",
            "industrial.parameter_out_of_range": "Paramètre hors de la plage sûre",
            "industrial.maintenance_required": "Maintenance requise",
            "industrial.calibration_needed": "Calibrage nécessaire",
            
            "status.online": "En ligne",
            "status.offline": "Hors ligne",
            "status.maintenance": "En maintenance",
            "status.degraded": "Performance dégradée",
            "status.operational": "Opérationnel"
        }
        
        # German (Germany)
        self.translations[SupportedLocale.DE_DE] = {
            "system.startup": "System wird gestartet...",
            "system.shutdown": "System wird heruntergefahren...",
            "system.ready": "System bereit",
            "system.error": "Systemfehler",
            "system.warning": "Systemwarnung",
            
            "auth.login": "Anmelden",
            "auth.logout": "Abmelden",
            "auth.unauthorized": "Nicht autorisierter Zugriff",
            "auth.invalid_credentials": "Ungültige Anmeldedaten",
            "auth.session_expired": "Sitzung abgelaufen",
            
            "safety.violation": "Sicherheitsverletzung erkannt",
            "safety.critical": "Kritischer Sicherheitsalarm",
            "safety.emergency_stop": "Not-Aus aktiviert",
            "security.threat_detected": "Sicherheitsbedrohung erkannt",
            "security.access_denied": "Zugriff verweigert",
            
            "industrial.process_started": "Industrieller Prozess gestartet",
            "industrial.process_stopped": "Industrieller Prozess gestoppt",
            "industrial.parameter_out_of_range": "Parameter außerhalb des sicheren Bereichs",
            "industrial.maintenance_required": "Wartung erforderlich",
            "industrial.calibration_needed": "Kalibrierung erforderlich",
            
            "status.online": "Online",
            "status.offline": "Offline",
            "status.maintenance": "In Wartung",
            "status.degraded": "Leistung beeinträchtigt",
            "status.operational": "Betriebsbereit"
        }
        
        # Japanese (Japan)
        self.translations[SupportedLocale.JA_JP] = {
            "system.startup": "システム起動中...",
            "system.shutdown": "システム終了中...",
            "system.ready": "システム準備完了",
            "system.error": "システムエラー",
            "system.warning": "システム警告",
            
            "auth.login": "ログイン",
            "auth.logout": "ログアウト",
            "auth.unauthorized": "不正アクセス",
            "auth.invalid_credentials": "認証情報が無効です",
            "auth.session_expired": "セッションが期限切れです",
            
            "safety.violation": "安全違反が検出されました",
            "safety.critical": "重要な安全アラート",
            "safety.emergency_stop": "緊急停止が作動しました",
            "security.threat_detected": "セキュリティ脅威が検出されました",
            "security.access_denied": "アクセスが拒否されました",
            
            "industrial.process_started": "産業プロセスが開始されました",
            "industrial.process_stopped": "産業プロセスが停止されました",
            "industrial.parameter_out_of_range": "パラメータが安全範囲外です",
            "industrial.maintenance_required": "メンテナンスが必要です",
            "industrial.calibration_needed": "キャリブレーションが必要です",
            
            "status.online": "オンライン",
            "status.offline": "オフライン",
            "status.maintenance": "メンテナンス中",
            "status.degraded": "性能低下",
            "status.operational": "運用中"
        }
        
        # Chinese Simplified (China)
        self.translations[SupportedLocale.ZH_CN] = {
            "system.startup": "系统启动中...",
            "system.shutdown": "系统关闭中...",
            "system.ready": "系统就绪",
            "system.error": "系统错误",
            "system.warning": "系统警告",
            
            "auth.login": "登录",
            "auth.logout": "登出",
            "auth.unauthorized": "未授权访问",
            "auth.invalid_credentials": "凭据无效",
            "auth.session_expired": "会话已过期",
            
            "safety.violation": "检测到安全违规",
            "safety.critical": "关键安全警报",
            "safety.emergency_stop": "紧急停止已激活",
            "security.threat_detected": "检测到安全威胁",
            "security.access_denied": "访问被拒绝",
            
            "industrial.process_started": "工业流程已启动",
            "industrial.process_stopped": "工业流程已停止",
            "industrial.parameter_out_of_range": "参数超出安全范围",
            "industrial.maintenance_required": "需要维护",
            "industrial.calibration_needed": "需要校准",
            
            "status.online": "在线",
            "status.offline": "离线",
            "status.maintenance": "维护中",
            "status.degraded": "性能下降",
            "status.operational": "运行中"
        }
        
    def set_locale(self, locale: SupportedLocale) -> bool:
        """Set the current locale."""
        try:
            if locale in self.translations:
                self.current_locale = locale
                self.logger.info(f"Locale changed to: {locale.value}")
                return True
            else:
                self.logger.warning(f"Locale {locale.value} not supported")
                return False
        except Exception as e:
            self.logger.error(f"Failed to set locale: {e}")
            return False
            
    def get_text(self, key: str, locale: Optional[SupportedLocale] = None, **kwargs) -> str:
        """Get localized text for given key."""
        target_locale = locale or self.current_locale
        
        try:
            # Try to get translation for target locale
            if target_locale in self.translations:
                translation = self.translations[target_locale].get(key)
                if translation:
                    # Apply string formatting if kwargs provided
                    if kwargs:
                        return translation.format(**kwargs)
                    return translation
                    
            # Fallback to default locale
            if self.default_locale in self.translations:
                fallback = self.translations[self.default_locale].get(key)
                if fallback:
                    self.logger.debug(f"Using fallback translation for key: {key}")
                    if kwargs:
                        return fallback.format(**kwargs)
                    return fallback
                    
            # If no translation found, return the key itself
            self.logger.warning(f"No translation found for key: {key}")
            return key
            
        except Exception as e:
            self.logger.error(f"Error getting translation for key {key}: {e}")
            return key
            
    def add_translation(self, locale: SupportedLocale, key: str, text: str) -> bool:
        """Add or update a translation."""
        try:
            if locale not in self.translations:
                self.translations[locale] = {}
                
            self.translations[locale][key] = text
            self.logger.debug(f"Added translation for {locale.value}: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add translation: {e}")
            return False
            
    def load_translations_from_file(self, file_path: str, locale: SupportedLocale) -> bool:
        """Load translations from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                translations = json.load(f)
                
            if locale not in self.translations:
                self.translations[locale] = {}
                
            self.translations[locale].update(translations)
            self.logger.info(f"Loaded translations from {file_path} for {locale.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load translations from {file_path}: {e}")
            return False
            
    def save_translations_to_file(self, file_path: str, locale: SupportedLocale) -> bool:
        """Save translations to JSON file."""
        try:
            if locale not in self.translations:
                self.logger.warning(f"No translations found for {locale.value}")
                return False
                
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.translations[locale], f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Saved translations to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save translations to {file_path}: {e}")
            return False
            
    def get_supported_locales(self) -> List[SupportedLocale]:
        """Get list of supported locales."""
        return list(self.translations.keys())
        
    def get_locale_info(self, locale: SupportedLocale) -> Dict[str, Any]:
        """Get information about a specific locale."""
        locale_names = {
            SupportedLocale.EN_US: {"name": "English (United States)", "native": "English"},
            SupportedLocale.EN_GB: {"name": "English (United Kingdom)", "native": "English"},
            SupportedLocale.ES_ES: {"name": "Spanish (Spain)", "native": "Español"},
            SupportedLocale.ES_MX: {"name": "Spanish (Mexico)", "native": "Español"},
            SupportedLocale.FR_FR: {"name": "French (France)", "native": "Français"},
            SupportedLocale.DE_DE: {"name": "German (Germany)", "native": "Deutsch"},
            SupportedLocale.JA_JP: {"name": "Japanese (Japan)", "native": "日本語"},
            SupportedLocale.ZH_CN: {"name": "Chinese (Simplified)", "native": "简体中文"},
            SupportedLocale.ZH_TW: {"name": "Chinese (Traditional)", "native": "繁體中文"},
            SupportedLocale.KO_KR: {"name": "Korean (South Korea)", "native": "한국어"},
            SupportedLocale.PT_BR: {"name": "Portuguese (Brazil)", "native": "Português"},
            SupportedLocale.IT_IT: {"name": "Italian (Italy)", "native": "Italiano"},
            SupportedLocale.RU_RU: {"name": "Russian (Russia)", "native": "Русский"},
            SupportedLocale.AR_SA: {"name": "Arabic (Saudi Arabia)", "native": "العربية"},
            SupportedLocale.HI_IN: {"name": "Hindi (India)", "native": "हिन्दी"}
        }
        
        info = locale_names.get(locale, {"name": locale.value, "native": locale.value})
        info.update({
            "code": locale.value,
            "available": locale in self.translations,
            "translation_count": len(self.translations.get(locale, {}))
        })
        
        return info
        
    def generate_translation_template(self, output_file: str) -> bool:
        """Generate translation template with all keys from default locale."""
        try:
            if self.default_locale not in self.translations:
                self.logger.error("Default locale translations not available")
                return False
                
            template = {}
            for key in self.translations[self.default_locale].keys():
                template[key] = f"[TRANSLATE: {self.translations[self.default_locale][key]}]"
                
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(template, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Translation template generated: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate translation template: {e}")
            return False
            
    def validate_translations(self, locale: SupportedLocale) -> Dict[str, Any]:
        """Validate translations for completeness and consistency."""
        if locale not in self.translations:
            return {"valid": False, "error": "Locale not found"}
            
        default_keys = set(self.translations[self.default_locale].keys())
        locale_keys = set(self.translations[locale].keys())
        
        missing_keys = default_keys - locale_keys
        extra_keys = locale_keys - default_keys
        
        # Check for placeholder translations
        placeholder_keys = []
        for key, value in self.translations[locale].items():
            if value.startswith("[TRANSLATE:") or value == key:
                placeholder_keys.append(key)
                
        completeness = (len(locale_keys) - len(placeholder_keys)) / len(default_keys) if default_keys else 0
        
        return {
            "valid": len(missing_keys) == 0 and len(placeholder_keys) == 0,
            "completeness": completeness,
            "missing_keys": list(missing_keys),
            "extra_keys": list(extra_keys),
            "placeholder_keys": placeholder_keys,
            "total_keys": len(locale_keys),
            "default_keys": len(default_keys)
        }


# Global localization manager instance
_localization_manager: Optional[LocalizationManager] = None


def get_localization_manager() -> LocalizationManager:
    """Get global localization manager instance."""
    global _localization_manager
    
    if _localization_manager is None:
        _localization_manager = LocalizationManager()
        
    return _localization_manager


def _(key: str, locale: Optional[SupportedLocale] = None, **kwargs) -> str:
    """Shortcut function for getting localized text."""
    return get_localization_manager().get_text(key, locale, **kwargs)


def set_locale(locale: SupportedLocale) -> bool:
    """Set global locale."""
    return get_localization_manager().set_locale(locale)


def get_current_locale() -> SupportedLocale:
    """Get current global locale."""
    return get_localization_manager().current_locale


# Example usage and testing
if __name__ == "__main__":
    # Initialize localization
    lm = LocalizationManager()
    
    # Test different locales
    test_locales = [SupportedLocale.EN_US, SupportedLocale.ES_ES, SupportedLocale.FR_FR, 
                   SupportedLocale.DE_DE, SupportedLocale.JA_JP, SupportedLocale.ZH_CN]
    
    test_key = "safety.critical"
    
    print("🌍 Localization Test Results:")
    print("=" * 50)
    
    for locale in test_locales:
        lm.set_locale(locale)
        translated = lm.get_text(test_key)
        locale_info = lm.get_locale_info(locale)
        print(f"{locale_info['native']:>12} ({locale.value}): {translated}")
        
    print("\n✅ Internationalization system ready for global deployment!")