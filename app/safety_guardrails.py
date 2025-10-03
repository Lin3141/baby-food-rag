from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .kg_retriever import KGSubgraph, KGFact
from .kg_query_parser import ParsedQuery

@dataclass
class SafetyBlock:
    food: str
    age_limit_months: int
    risk_type: str
    reason: str
    source: str
    severity: str  # "CRITICAL", "WARNING", "INFO"

class SafetyGuardrailEngine:
    def __init__(self):
        # Hard-coded critical safety blocks that AI must NEVER override
        self.critical_safety_blocks = [
            SafetyBlock("Honey", 12, "botulism", "risk of infant botulism", "AAP Guidelines", "CRITICAL"),
            SafetyBlock("Whole grapes", 48, "choking", "choking hazard", "AAP Choking Guidelines", "CRITICAL"),
            SafetyBlock("Whole nuts", 48, "choking", "choking hazard", "AAP Choking Guidelines", "CRITICAL"),
            SafetyBlock("Cow's milk", 12, "anemia", "anemia risk if used as primary drink", "AAP Guidelines", "CRITICAL"),
            SafetyBlock("Shellfish", 6, "allergy", "high allergy risk - introduce carefully", "AAP Allergy Guidelines", "WARNING"),
        ]
        
        # Dynamic safety rules extracted from KG
        self.dynamic_safety_rules = {}
    
    def check_safety_violations(self, parsed_query: ParsedQuery, subgraph: KGSubgraph) -> Optional[SafetyBlock]:
        """Check if query violates any critical safety rules"""
        
        if not parsed_query.food or not parsed_query.age_months:
            return None
        
        # Check critical hard-coded blocks first
        violated_block = self._check_critical_blocks(parsed_query)
        if violated_block:
            return violated_block
        
        # Check KG-derived safety blocks
        kg_violation = self._check_kg_safety_blocks(parsed_query, subgraph)
        if kg_violation:
            return kg_violation
        
        return None
    
    def _check_critical_blocks(self, parsed_query: ParsedQuery) -> Optional[SafetyBlock]:
        """Check against hard-coded critical safety blocks"""
        food_lower = parsed_query.food.lower()
        age = parsed_query.age_months
        
        for block in self.critical_safety_blocks:
            if block.food.lower() in food_lower or food_lower in block.food.lower():
                if age < block.age_limit_months:
                    return block
        return None
    
    def _check_kg_safety_blocks(self, parsed_query: ParsedQuery, subgraph: KGSubgraph) -> Optional[SafetyBlock]:
        """Check KG facts for safety violations"""
        age = parsed_query.age_months
        food = parsed_query.food
        
        # Check age restrictions from KG
        for fact in subgraph.facts:
            if fact.relation == "SAFE_AT":
                min_safe_age = int(fact.object)
                if age < min_safe_age:
                    return SafetyBlock(
                        food=food,
                        age_limit_months=min_safe_age,
                        risk_type="age_restriction",
                        reason=f"not recommended for babies under {min_safe_age} months",
                        source=fact.source,
                        severity="WARNING"
                    )
        
        # Check for critical risks
        for fact in subgraph.facts:
            if fact.relation == "HAS_RISK":
                risk = fact.object.lower()
                if risk == "botulism":
                    return SafetyBlock(
                        food=food,
                        age_limit_months=12,
                        risk_type="botulism",
                        reason="risk of infant botulism",
                        source=fact.source,
                        severity="CRITICAL"
                    )
                elif risk == "choking" and age < 12:
                    return SafetyBlock(
                        food=food,
                        age_limit_months=age,
                        risk_type="choking",
                        reason="choking hazard for young babies",
                        source=fact.source,
                        severity="WARNING"
                    )
        
        return None
    
    def generate_safety_block_response(self, violation: SafetyBlock, parsed_query: ParsedQuery) -> str:
        """Generate structured safety block response"""
        if violation.severity == "CRITICAL":
            safety_emoji = "❌"
            verdict = "Not safe"
        else:
            safety_emoji = "⚠️"
            verdict = "Caution required"
        
        response_parts = [
            f"{safety_emoji} {verdict}. Babies under {violation.age_limit_months} months should not consume {violation.food.lower()} ({violation.reason})."
        ]
        
        # Add specific guidance
        if violation.risk_type == "botulism":
            response_parts.append("🚨 Critical: This is a serious safety concern - never give honey to infants.")
        elif violation.risk_type == "choking":
            response_parts.append("🔄 Alternative: Wait until appropriate age or modify preparation.")
        elif violation.risk_type == "anemia":
            response_parts.append("🥛 Note: Small amounts in food are okay, but not as primary drink.")
        
        # Always include authoritative source
        response_parts.append(f"📚 Sources: {violation.source}")
        
        return "\n".join(response_parts)
