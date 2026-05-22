from typing import List, Dict, TypedDict, Optional

class RoleRequirements(TypedDict):
    description: str
    expectations: List[str]

class PersonaRole:
    def __init__(self, name: str, role_type: str, requirements: RoleRequirements):
        self.name = name
        self.role_type = role_type  # e.g., 'performer' or 'target_audience'
        self.requirements = requirements

class PersonaFramework:
    def __init__(self):
        self.roles: Dict[str, PersonaRole] = {}
        self.categories: List[str] = []

    def add_role(self, role: PersonaRole):
        self.roles[role.name] = role

    def get_role(self, name: str) -> Optional[PersonaRole]:
        return self.roles.get(name)

    def list_roles(self) -> List[str]:
        return list(self.roles.keys())

    def add_category(self, category: str):
        if category not in self.categories:
            self.categories.append(category)

    def list_categories(self) -> List[str]:
        return self.categories

# Define performer roles
performer_roles = [
    PersonaRole(
        name="Sales Performer",
        role_type="performer",
        requirements={
            "description": "Role responsible for performing sales calls and client interactions.",
            "expectations": [
                "Effectively introduce self and company.",
                "Understand client persona and needs.",
                "Adapt communication style based on client profile.",
                "Achieve lead generation and product mapping goals."
            ]
        }
    ),
    PersonaRole(
        name="Customer Support",
        role_type="performer",
        requirements={
            "description": "Role responsible for supporting customers post-sale.",
            "expectations": [
                "Provide timely and accurate information.",
                "Resolve customer issues efficiently.",
                "Maintain positive customer relationships."
            ]
        }
    )
]

# Define target audience/opponent roles
target_roles = [
    PersonaRole(
        name="CEO of SME Company",
        role_type="target_audience",
        requirements={
            "description": "Small and Medium Enterprise CEO with decision-making power.",
            "expectations": [
                "Interested in ROI and business growth.",
                "Values concise and data-driven communication.",
                "Prefers strategic discussions."
            ]
        }
    ),
    PersonaRole(
        name="DINK working in IT firm",
        role_type="target_audience",
        requirements={
            "description": "Dual Income No Kids professional in IT sector.",
            "expectations": [
                "Appreciates technical details.",
                "Prefers efficient and clear communication.",
                "Open to innovative solutions."
            ]
        }
    ),
    PersonaRole(
        name="Retired Government Officer",
        role_type="target_audience",
        requirements={
            "description": "Retired officer with experience in government sector.",
            "expectations": [
                "Values trust and reliability.",
                "Prefers formal communication.",
                "Interested in community impact."
            ]
        }
    ),
    PersonaRole(
        name="School Teacher still active",
        role_type="target_audience",
        requirements={
            "description": "Active school teacher with interest in educational products.",
            "expectations": [
                "Seeks practical benefits.",
                "Prefers clear and simple explanations.",
                "Values empathy and patience."
            ]
        }
    ),
    PersonaRole(
        name="Mid-level Manager with 2 kids",
        role_type="target_audience",
        requirements={
            "description": "Manager balancing work and family responsibilities.",
            "expectations": [
                "Needs flexible solutions.",
                "Prefers time-saving communication.",
                "Interested in family-oriented benefits."
            ]
        }
    )
]

# Define practice session categories
practice_categories = [
    "Lead Generation",
    "Connecting",
    "Profiling",
    "Needs Gathering",
    "Product Mapping"
]

# Initialize framework and add roles and categories
persona_framework = PersonaFramework()

for role in performer_roles + target_roles:
    persona_framework.add_role(role)

for category in practice_categories:
    persona_framework.add_category(category)
