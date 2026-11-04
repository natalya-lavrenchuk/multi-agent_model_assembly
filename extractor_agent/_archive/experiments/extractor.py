import fitz
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# Flattened Schema
class ExtractedQuantity(BaseModel):
    symbol: str
    name_or_description: str
    quantity_type: Literal["StateQuantity", "RateQuantity", "Parameter", "Constant", "ControlQuantity"]
    unit_string: Optional[str] = Field(None, description="e.g., g/L, h^-1")
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    verbatim_quote: str

class QuantityHarvestPayload(BaseModel):
    quantities: List[ExtractedQuantity]

# Extract only Page 5 (Index 4) 
doc = fitz.open("statistical_tools.pdf")
pdf_text = doc[4].get_text() 

client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON
)

prompt = "Extract all physical quantities, state variables, parameters, and controls. Extract the local symbol verbatim."

print("Sending text to Llama 3.1... (This may take 30-90 seconds depending on your Mac's chip)")

response = client.chat.completions.create(
    model="llama3.1",
    response_model=QuantityHarvestPayload,
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Text:\n\n{pdf_text}"}
    ]
)

print("--- EXTRACTION COMPLETE ---")
for q in response.quantities:
    print(f"{q.symbol} ({q.quantity_type}): {q.name_or_description} [{q.unit_string}]")
