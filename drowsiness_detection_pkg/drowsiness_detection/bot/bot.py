#!/usr/bin/env python3

"""This module contains the driver assistance bot which takes the input such as driver eye
blink frequency, PERCLOS, Yawning frequency and alert the driver if he/she gets drowsy.
"""

from typing import (
    List,
    Optional,
    Dict,
    Sequence,
    Callable,
    Any,
)
from pydantic import BaseModel, Field, StrictFloat, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


class Bot:
    """Bot schemas"""

    class BotConfig(BaseModel):
        """Configuration for the driver assistance bot."""

        model_id: str
        system_prompt: str | None = None
        user_prompt: str | None = None
        output_schema: Dict[str, Any] | None = Field(
            default=None,
            description="Pydantic schema class for structured output (e.g., Output)",
        )
        tools: Sequence[Callable[..., Any]] | None = None
        temperature: float | None = None

    class Input(BaseModel):
        """Input data for the driver assistance bot"""

        perclos: Optional[StrictFloat] = Field(
            default=None, description="Percentage of time eyes are closed."
        )
        blink_rate: Optional[StrictFloat] = Field(
            default=None, description="Number of eye blinks per minute."
        )
        yawn_freq: Optional[StrictFloat] = Field(
            default=None, description="Number of yawns per minute."
        )
        sdlp: Optional[StrictFloat] = Field(
            default=None, description="Standard deviation of lane position (m)."
        )
        steering_entropy: Optional[StrictFloat] = Field(
            default=None, description="Unpredictability measure of steering movements."
        )
        steering_reversal_rate: Optional[StrictFloat] = Field(
            default=None, description="Steering direction changes per minute."
        )

    class ToolCall(BaseModel):
        """Schema for each tool call."""

        name: str = Field(description="The name of the tool to use.")
        args: Dict[str, Any] = Field(description="Arguments required for the tool.")
        description: str = Field(description="Brief explanation of what the tool does.")

    class Output(BaseModel):
        """Output class without actions for driver assistance bot"""

        drowsiness_level: str = Field(description="Detected drowsiness risk level.")
        reasoning: str = Field(
            description="Explanation based on input metrics that led to the decision."
        )
        tool_calls: List["Bot.ToolCall"] = Field(
            description="List of tool calls to execute in response to detected drowsiness."
        )


class BaseBot(Bot):
    """Base class for the driver assistance bot.

    Args:
        Bot (Bot): The base bot class.
    """

    def __init__(
        self,
        config: Bot.BotConfig,
    ):
        self.llm = ChatOllama(
            model=config.model_id,
            temperature=config.temperature,
        )
        # prompts
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", config.system_prompt),
                ("user", config.user_prompt),
            ]
        )
        # Bind tools and structured output
        llm_with_tools = self.llm.bind_tools(config.tools)
        self.chain = prompt | llm_with_tools

    def invoke(self, input_data: Bot.Input) -> Bot.Output:
        """Invoke the driver assistance bot.

        Args:
            input_data (Bot.Input): Input data for the bot.

        Raises:
            e: ValidationError if the output is not in the structured format.

        Returns:
            Bot.Output: The structured output from the bot.
        """
        # Format the input data
        response = self.chain.invoke({"drowsiness_metrics": input_data.dict()})
        # print("Raw response:", response.content)
        try:
            validated_output = Bot.Output.model_validate_json(response.content)
            return validated_output
        except ValidationError as e:
            print("Validation failed:", e)
            raise e
