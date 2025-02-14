from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


logger = logging.getLogger(__name__)


class EquivalenceInput(BaseModel):
    reference: str = Field(..., description="Reference SPARQL query")
    response: str = Field(..., description="Generated SPARQL query")
    rdf_schema: str = Field(..., description="Reference RDF schema in Turtle format")


class EquivalenceOutput(BaseModel):
    response_query_explanation: str = Field(
        ..., description="Explanation of the generated SPARQL query"
    )
    reference_query_explanation: str = Field(
        ..., description="Explanation of the reference SPARQL query"
    )
    equivalence: bool = Field(
        ...,
        description="Whether the generated SPARQL query is equivalent to the reference SPARQL query",
    )


class EquivalencePrompt(PydanticPrompt[EquivalenceInput, EquivalenceOutput]):
    instruction = """
    Explain and compare two SPARQL queries (Q1 and Q2) based on the provided RDF schema. First, explain each query, then determine if they have significant logical differences.
    """
    input_model = EquivalenceInput
    output_model = EquivalenceOutput
    examples = [
        (
            EquivalenceInput(
                reference="""
                    PREFIX ex: <http://example.org/>
                    SELECT ?name WHERE {
                        ?user a ex:User .
                        ?user ex:active "true"^^xsd:boolean .
                        ?user ex:name ?name .
                    }
                """,
                response="""
                    PREFIX ex: <http://example.org/>
                    SELECT ?name WHERE {
                        ?user a ex:User .
                        ?user ex:active "1"^^xsd:integer .
                        ?user ex:name ?name .
                    }
                """,
                rdf_schema="""
                    @prefix ex: <http://example.org/> .
                    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
                    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
                    
                    ex:User a rdfs:Class .
                    
                    ex:active a rdf:Property ;
                        rdfs:domain ex:User ;
                        rdfs:range xsd:boolean .
                    
                    ex:name a rdf:Property ;
                        rdfs:domain ex:User ;
                        rdfs:range xsd:string .
                    
                    ex:User1 a ex:User ;
                        ex:active "true"^^xsd:boolean ;
                        ex:name "Alice" .
                    
                    ex:User2 a ex:User ;
                        ex:active "1"^^xsd:integer ;
                        ex:name "Bob" .
                """,
            ),
            EquivalenceOutput(
                response_query_explanation="The generated SPARQL query selects the names of users where the 'active' property has a value of '1' as an integer.",
                reference_query_explanation="The reference SPARQL query selects the names of users where the 'active' property is explicitly a boolean with value 'true'.",
                equivalence=True,
            ),
        )
    ]


@dataclass
class LLMSPARQLEquivalence(MetricWithLLM, SingleTurnMetric):
    name: str = "llm_sparql_equivalence_with_reference"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"response", "reference", "reference_contexts"}
        }
    )
    output_type: t.Optional[MetricOutputType] = MetricOutputType.BINARY
    equivalence_prompt: PydanticPrompt = EquivalencePrompt()

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not initialized"
        assert isinstance(sample.reference, str), "Sample reference must be a string"
        assert isinstance(sample.response, str), "Sample response must be a string"
        assert isinstance(
            sample.reference_contexts, list
        ), "Sample reference_contexts must be a List"

        rdf_schema = "\n".join(sample.reference_contexts)
        input_data = EquivalenceInput(
            reference=sample.reference,
            response=sample.response,
            rdf_schema=rdf_schema,
        )
        response = await self.equivalence_prompt.generate(
            data=input_data, llm=self.llm, callbacks=callbacks
        )
        return int(response.equivalence)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)
