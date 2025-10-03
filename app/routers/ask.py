from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from ..models import AskRequest, AskResponse, Citation
from ..retriever import HybridRetriever
from ..kg_query_parser import BabyFoodQueryParser, ParsedQuery
from ..kg_retriever import KnowledgeGraphRetriever, KGSubgraph
from ..safety_guardrails import SafetyGuardrailEngine

router = APIRouter()

# Global retriever instance (will be set in main.py)
retriever: HybridRetriever = None

def get_retriever():
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    return retriever

@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, ret: HybridRetriever = Depends(get_retriever)):
    """
    Ask a question about baby foods using Knowledge Graph RAG flow with safety guardrails
    """
    try:
        # Step 1: Parse query to extract food + age
        food_names = [food.name for food in ret.foods]
        parser = BabyFoodQueryParser(food_names)
        parsed_query = parser.parse_query(request.question)
        
        # Step 2: Retrieve subgraph from KG
        kg_retriever = KnowledgeGraphRetriever(ret.foods)
        subgraph = kg_retriever.retrieve_subgraph(parsed_query)
        
        # Step 3: SAFETY GUARDRAILS CHECK - This is critical and cannot be overridden
        safety_engine = SafetyGuardrailEngine()
        safety_violation = safety_engine.check_safety_violations(parsed_query, subgraph)
        
        if safety_violation:
            # HARD BLOCK: Return safety violation response immediately
            safety_response = safety_engine.generate_safety_block_response(safety_violation, parsed_query)
            
            citations = [
                Citation(
                    food_name=parsed_query.food or "Safety Guidelines",
                    usda_url=f"https://www.healthychildren.org/English/ages-stages/baby/feeding-nutrition/Pages/Starting-Solid-Foods.aspx",
                    relevance_score=1.0
                )
            ]
            
            return AskResponse(
                answer=safety_response,
                citations=citations,
                confidence="High",  # High confidence on safety blocks
                retrieved_foods=[]
            )
        
        # Step 4: Continue with normal flow only if no safety violations
        if not subgraph.facts:
            # Fallback to traditional retrieval if no KG facts found
            retrieved_foods, scores = ret.retrieve(request.question, request.top_k)
            answer, confidence = ret.generate_answer(request.question, retrieved_foods, scores)
            citations = ret.get_citations(retrieved_foods, scores)
            
            return AskResponse(
                answer=answer,
                citations=citations,
                confidence=confidence,
                retrieved_foods=retrieved_foods
            )
        
        # Step 3: Generate LLM prompt with facts
        llm_prompt = kg_retriever.generate_llm_prompt(parsed_query, subgraph)
        
        # Step 4: Generate structured answer (simplified LLM simulation)
        answer = simulate_llm_response(parsed_query, subgraph)
        
        # Step 5: Generate citations from KG facts
        citations = [
            Citation(
                food_name=parsed_query.food or "Unknown",
                usda_url=f"https://fdc.nal.usda.gov/search?query={parsed_query.food or 'food'}",
                relevance_score=1.0
            )
        ]
        
        # Determine confidence based on safety flags
        internal_confidence = "High"
        if subgraph.safety_flags:
            if any("too_young" in flag for flag in subgraph.safety_flags):
                internal_confidence = "Low"
            else:
                internal_confidence = "Medium"
        
        # Convert to parent-friendly confidence indicator
        confidence_display = _get_parent_friendly_confidence(internal_confidence, subgraph)
        
        return AskResponse(
            answer=answer,
            citations=citations,
            confidence=confidence_display,  # Parent-friendly version
            retrieved_foods=retrieved_foods
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

def simulate_llm_response(parsed_query: ParsedQuery, subgraph: KGSubgraph) -> str:
    """Generate simplified, scannable response with clear visual hierarchy"""
    if not parsed_query.food:
        return "❌ **No food identified** in your question\n**Source:** Query Parser"
    
    response_parts = []
    food_name = parsed_query.food.lower()
    
    # 1. SAFETY VERDICT (✅/❌ + bold)
    age_facts = [f for f in subgraph.facts if f.relation == "SAFE_AT"]
    risk_facts = [f for f in subgraph.facts if f.relation == "HAS_RISK"]
    
    if any("too_young" in flag for flag in subgraph.safety_flags):
        response_parts.append(f"❌ **Wait until {parsed_query.age_months + 2}+ months** for {food_name}")
    elif age_facts:
        min_age = age_facts[0].object
        response_parts.append(f"✅ **Yes, {food_name} is safe** from {min_age} months")
    else:
        response_parts.append(f"✅ **Yes, {food_name} is safe** for babies")
    
    # 2. WHY IT MATTERS (reasoning without icon)
    why_explanation = _get_why_it_matters(food_name, parsed_query.age_months)
    if why_explanation:
        response_parts.append(f"\n**Why it matters:** {why_explanation}")
    
    # 3. PREPARATION (spacing without icon)
    prep_instruction = _get_simple_prep(food_name)
    if prep_instruction:
        response_parts.append(f"\n**Prep:** {prep_instruction}")
    
    # 4. KEY WARNING (bold without icon)
    warning = _get_key_warning(risk_facts, food_name)
    if warning:
        response_parts.append(f"\n**Note:** {warning}")
    
    # 5. BENEFIT (bold without icon)
    benefit = _get_key_benefit(food_name, subgraph.facts)
    if benefit:
        response_parts.append(f"\n**Benefit:** {benefit}")
    
    # 6. ACTIONABLE NEXT STEP (practical action without icon)
    action_step = _get_actionable_next_step(food_name, parsed_query.age_months, risk_facts)
    if action_step:
        # Remove icons from action step
        clean_action = action_step.replace("👍 **Next step:**", "**Next step:**").replace("🚫 **Next step:**", "**Next step:**").replace("⚠️ **Next step:**", "**Next step:**")
        response_parts.append(f"\n{clean_action}")
    
    # 7. SOURCE (clean formatting without icon)
    source = _get_primary_source(subgraph.facts)
    response_parts.append(f"\n**Sources:** {source}")
    
    return "".join(response_parts)

def _get_actionable_next_step(food_name: str, age_months: Optional[int], risk_facts: List) -> str:
    """Generate practical next step for parents"""
    
    # Check for safety blocks first
    risks = [fact.object for fact in risk_facts] if risk_facts else []
    
    if 'botulism' in risks:
        return "🚫 **Next step:** Avoid before 12 months. Try again after first birthday."
    
    if any("too_young" in str(risk) for risk in risks):
        return f"🚫 **Next step:** Wait a few more months, then try again."
    
    # Age-appropriate action steps
    action_map = {
        'banana': {
            6: "👍 **Next step:** Perfect first food. Try mashed banana mixed into baby cereal.",
            8: "👍 **Next step:** Cut into soft strips for self-feeding practice.",
            10: "👍 **Next step:** Try small banana pieces as finger food."
        },
        'apple': {
            6: "👍 **Next step:** Steam and mash smooth. Mix with a familiar food like rice cereal.",
            8: "👍 **Next step:** Try soft cooked apple pieces for texture practice.",
            10: "👍 **Next step:** Offer as soft finger food pieces."
        },
        'avocado': {
            6: "👍 **Next step:** Mash ripe avocado and offer as first food. Mix with breast milk if needed.",
            8: "👍 **Next step:** Cut into soft strips for baby-led weaning.",
            10: "👍 **Next step:** Perfect finger food - cut into small cubes."
        },
        'sweet potato': {
            6: "👍 **Next step:** Steam until very soft, mash smooth. Great first vegetable choice.",
            8: "👍 **Next step:** Try soft roasted sweet potato sticks.",
            10: "👍 **Next step:** Cut into small cubes for finger feeding."
        },
        'chicken': {
            6: "👍 **Next step:** Cook thoroughly and puree with water or breast milk.",
            8: "👍 **Next step:** Shred finely and mix with favorite vegetables.",
            10: "👍 **Next step:** Try small, soft shredded pieces as finger food."
        },
        'salmon': {
            6: "👍 **Next step:** Cook well, flake carefully (check for bones), and puree smooth.",
            8: "👍 **Next step:** Flake into small pieces, mix with vegetables.",
            10: "👍 **Next step:** Offer small flakes as finger food (always check for bones)."
        },
        'egg': {
            6: "👍 **Next step:** Start with well-scrambled egg, mashed smooth.",
            8: "👍 **Next step:** Try soft scrambled egg pieces.",
            10: "👍 **Next step:** Perfect finger food - cut scrambled egg into small pieces."
        },
        'rice cereal': {
            6: "👍 **Next step:** Mix thin with breast milk or formula. Perfect first food choice.",
            8: "👍 **Next step:** Make thicker consistency as baby gets used to textures.",
            10: "👍 **Next step:** Mix with fruit purees for variety."
        },
        'spinach': {
            6: "👍 **Next step:** Steam and puree finely. Start with small portions mixed with other foods.",
            8: "👍 **Next step:** Mix chopped spinach into familiar foods like pasta or rice.",
            10: "👍 **Next step:** Try soft cooked spinach pieces."
        },
        'yogurt': {
            6: "👍 **Next step:** Offer plain, whole-milk yogurt. Great for breakfast or snack.",
            8: "👍 **Next step:** Mix with fruit purees for natural sweetness.",
            10: "👍 **Next step:** Perfect finger food with soft fruit pieces."
        }
    }
    
    # Get age-appropriate action
    if food_name in action_map and age_months:
        if age_months >= 10:
            return action_map[food_name].get(10, action_map[food_name].get(8, action_map[food_name].get(6)))
        elif age_months >= 8:
            return action_map[food_name].get(8, action_map[food_name].get(6))
        else:
            return action_map[food_name].get(6)
    
    # Generic actions based on food type
    if 'choking' in risks:
        return "⚠️ **Next step:** Prepare safely - avoid large pieces. Always supervise eating."
    
    # Default actions by food category
    if any(fruit in food_name for fruit in ['apple', 'pear', 'peach']):
        return "👍 **Next step:** Steam until soft, mash smooth for first tries."
    elif any(veg in food_name for veg in ['carrot', 'broccoli', 'peas']):
        return "👍 **Next step:** Steam until very soft, start with puree consistency."
    elif any(protein in food_name for protein in ['chicken', 'turkey', 'beef']):
        return "👍 **Next step:** Cook thoroughly, puree or shred finely to start."
    else:
        return "👍 **Next step:** Start with small portions mixed with familiar foods."

def _get_why_it_matters(food_name: str, age_months: Optional[int]) -> str:
    """Explain why this food recommendation matters for parents"""
    explanations = {
        'apple': 'Apples are recommended as first foods because they\'re naturally sweet, easy to digest, and high in fiber for healthy digestion',
        'banana': 'Bananas are recommended as a first food because they\'re soft, easy to digest, and packed with potassium and fiber',
        'avocado': 'Avocados provide healthy fats essential for brain development during your baby\'s rapid growth phase',
        'sweet potato': 'Sweet potatoes are ideal first foods - naturally sweet, soft when cooked, and loaded with vitamin A for healthy vision',
        'rice cereal': 'Iron-fortified cereals help prevent iron deficiency, which is common in babies after 6 months as iron stores from birth deplete',
        'spinach': 'Leafy greens like spinach provide iron and folate, but small portions are recommended for young babies due to nitrate content',
        'chicken': 'Chicken provides complete protein with all essential amino acids needed for your baby\'s rapid growth and development',
        'salmon': 'Salmon offers omega-3 fatty acids crucial for brain and eye development during the first year',
        'egg': 'Eggs are a complete protein source and early introduction may actually help prevent egg allergies later',
        'yogurt': 'Plain yogurt introduces beneficial bacteria for gut health while providing protein and calcium for bone development',
        'broccoli': 'Broccoli is packed with vitamin C and folate, supporting immune system development and healthy growth',
        'lentils': 'Lentils provide plant-based protein and iron, offering variety in protein sources for growing babies',
        'honey': 'Honey is avoided before 12 months because babies\' immune systems can\'t fight botulism spores that may be present',
        'whole grapes': 'Whole grapes are a choking hazard due to their size and firm texture - always quarter them lengthwise',
        'peanut': 'Early peanut introduction (around 6 months) may actually reduce the risk of developing peanut allergies'
    }
    
    # Check for specific food explanations
    for food_key, explanation in explanations.items():
        if food_key in food_name:
            return explanation
    
    # Age-specific reasoning
    if age_months and age_months < 8:
        return f'{food_name.title()} is appropriate for your baby\'s developmental stage and helps introduce new flavors and textures'
    elif age_months and age_months >= 8:
        return f'At {age_months} months, your baby can handle more complex textures like {food_name}'
    
    return f'{food_name.title()} provides important nutrients during your baby\'s critical growth period'

def _get_key_benefit(food_name: str, facts: List) -> str:
    """Get parent-friendly benefit instead of technical numbers"""
    benefit_map = {
        'apple': 'Rich in fiber for healthy digestion',
        'banana': 'Good source of potassium for muscle growth',
        'avocado': 'Healthy fats for brain development',
        'sweet potato': 'Supports healthy vision development',
        'spinach': 'Iron-rich for healthy blood',
        'broccoli': 'Provides vitamin C to support immunity',
        'chicken': 'Complete protein for muscle growth',
        'salmon': 'Omega-3s for brain development',
        'egg': 'Complete protein plus brain-building choline',
        'rice cereal': 'Iron-fortified to support healthy blood',
        'yogurt': 'Probiotics for healthy gut',
        'lentils': 'Plant protein for growth and development'
    }
    
    for food_key, benefit in benefit_map.items():
        if food_key in food_name:
            return benefit
    
    # Check KG facts for nutrients and convert to benefits
    nutrient_facts = [f for f in facts if f.relation == "CONTAINS"]
    if nutrient_facts:
        nutrient = nutrient_facts[0].object
        if nutrient == "iron":
            return "Supports healthy blood development"
        elif nutrient == "vitamin_c":
            return "Provides vitamin C to support immunity"
        elif nutrient == "vitamin_a":
            return "Supports healthy vision development"
        elif nutrient == "protein":
            return "Supports muscle growth and development"
    
    return ""

def _get_simple_prep(food_name: str) -> str:
    """Get simplified, action-focused preparation instructions"""
    prep_map = {
        'apple': 'Steam and mash, then soft pieces as baby grows',
        'banana': 'Mash or cut into soft strips', 
        'pear': 'Steam and mash for beginners',
        'avocado': 'Mash or cut into strips',
        'sweet potato': 'Steam until very soft, then mash',
        'carrot': 'Steam until very soft, never raw',
        'broccoli': 'Steam soft, serve small florets',
        'spinach': 'Steam and puree finely',
        'chicken': 'Cook thoroughly, shred finely',
        'salmon': 'Cook well, check for bones carefully',
        'egg': 'Scramble well-cooked',
        'rice cereal': 'Mix thin with breastmilk or formula',
        'oatmeal': 'Cook soft, thin consistency',
        'yogurt': 'Serve plain, full-fat',
        'cheese': 'Shred finely or melt',
        'lentils': 'Cook very soft, mash well'
    }
    
    # Check for exact matches
    for food_key, instruction in prep_map.items():
        if food_key in food_name:
            return instruction
    
    # Fallback based on food type
    if any(fruit in food_name for fruit in ['berry', 'fruit']):
        return 'Mash or cut small for young babies'
    elif any(veg in food_name for veg in ['vegetable', 'green']):
        return 'Steam until very soft'
    elif any(protein in food_name for protein in ['meat', 'fish']):
        return 'Cook thoroughly, serve in small pieces'
    
    return ""

def _get_key_warning(risk_facts: List, food_name: str) -> str:
    """Get the most important warning in simple terms"""
    if not risk_facts:
        return ""
    
    # Prioritize warnings by severity
    risks = [fact.object for fact in risk_facts]
    
    if 'choking' in risks:
        if 'grape' in food_name:
            return 'Cut grapes in quarters lengthwise'
        elif 'nut' in food_name:
            return 'Never give whole nuts before age 4'
        else:
            return 'Avoid large chunks (choking risk)'
    elif 'botulism' in risks:
        return 'Never give to babies under 12 months'
    elif 'allergy' in risks:
        return 'Watch for allergic reactions'
    elif 'nitrate' in risks:
        return 'Small portions only for young babies'
    
    return risks[0].replace('_', ' ').title() + ' risk'

def _get_primary_source(facts: List) -> str:
    """Get the most authoritative source without redundancy"""
    sources = [fact.source for fact in facts]
    
    # Prioritize medical authorities
    if any('AAP' in source for source in sources):
        return 'AAP Guidelines'
    elif any('CDC' in source for source in sources):
        return 'CDC Guidelines' 
    elif any('WHO' in source for source in sources):
        return 'WHO Guidelines'
    
    return 'Pediatric Guidelines'

def _get_parent_friendly_confidence(internal_confidence: str, subgraph) -> str:
    """Convert internal confidence to parent-friendly indicator"""
    
    # Check if we have medical authority sources
    has_medical_sources = any(
        any(authority in fact.source for authority in ['AAP', 'CDC', 'WHO'])
        for fact in subgraph.facts
    )
    
    if internal_confidence == "High":
        if has_medical_sources:
            return "Backed by medical guidelines"
        else:
            return "Well-established guidance"
    elif internal_confidence == "Medium":
        return "General guidance available"
    else:  # Low confidence
        return "Limited guidance - consult pediatrician"
