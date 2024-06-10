from typing import Any
from langchain.agents import tool
from langserve.pydantic_v1 import BaseModel, Field

class TreatmentInput(BaseModel):
    treatment: str = Field(description="name of the treatment")

@tool("get_treatment_price", args_schema=TreatmentInput)
def get_treatment_price(
    treatment: str,
    **kwargs: Any,
):
    """
    Return cost of a food
    """
    print("treatment", treatment)
    print("kwargs", kwargs)
    return "100 rs"