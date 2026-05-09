# Copyright (c) 2026 CortexFlow / Adrian Creox. All rights reserved.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import google.generativeai as genai
from typing import Any
import structlog

from cortexflow.providers.base import (
    LLMProvider, 
    Completion, 
    CompletionRequest, 
    Message, 
    ProviderError,
    ToolCall
)

logger = structlog.get_logger(__name__)

class GeminiProvider(LLMProvider):
    """
    Google Gemini Adapter for CortexFlow.
    Optimized for Gemini 1.5 Pro/Flash with native tool-calling.
    """

    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        self._api_key = api_key
        self._model_name = model
        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(model_name=model)

    @property
    def name(self) -> str:
        return f"google-{self._model_name}"

    async def complete(self, request: CompletionRequest) -> Completion:
        """Translates CortexFlow request to Gemini API and back."""
        try:
            # 1. Mapeo de herramientas (Tools)
            tools = self._prepare_tools(request.tools) if request.tools else None
            
            # 2. Conversión de mensajes al formato de Google
            contents = self._transform_messages(request.messages)
            
            # 3. Inferencia
            # Nota: Gemini usa un sistema de 'chat session' o 'generate_content'
            response = await self._client.generate_content_async(
                contents,
                tools=tools,
                generation_config={
                    "temperature": request.temperature,
                    "max_output_tokens": request.max_tokens,
                }
            )

            # 4. Extracción de resultados
            return self._parse_response(response)

        except Exception as e:
            logger.error("gemini.completion.failed", error=str(e))
            raise ProviderError(f"Gemini failed: {str(e)}", provider=self.name)

    def _transform_messages(self, messages: list[Message]) -> list[dict]:
        """Convierte mensajes de CortexFlow a Gemini 'contents'."""
        gemini_msgs = []
        for m in messages:
            # Gemini separa 'system' instructions en el constructor o como rol 'user' inicial
            # Para simplificar y mantener compatibilidad:
            role = "user" if m.role in ["user", "system"] else "model"
            gemini_msgs.append({
                "role": role,
                "parts": [m.content]
            })
        return gemini_msgs

    def _prepare_tools(self, tools: list[Any]) -> list[dict]:
        """Mapea ToolSchema de CortexFlow a Gemini Function Declarations."""
        declarations = []
        for t in tools:
            declarations.append({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            })
        return [{"function_declarations": declarations}]

    def _parse_response(self, response: Any) -> Completion:
        """Extrae texto o llamadas a herramientas de la respuesta de Gemini."""
        content = ""
        tool_calls = []
        
        # Gemini puede devolver múltiples 'parts'
        for part in response.candidates[0].content.parts:
            if fn := part.function_call:
                tool_calls.append(ToolCall(
                    call_id=f"gemini-{fn.name}", # Gemini no siempre da un ID, generamos uno
                    tool_name=fn.name,
                    arguments=dict(fn.args)
                ))
            elif text := part.text:
                content += text

        # Token tracking (safe fallback)
        usage = getattr(response, "usage_metadata", None)
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0

        return Completion(
            content=content if content else None,
            tool_calls=tool_calls,
            model=self._model_name,
            stop_reason="tool_use" if tool_calls else "stop",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


    async def health(self) -> bool:
        # Check simple para verificar conectividad (usando un modelo pequeño)
        try:
            await self._client.generate_content_async("ping")
            return True
        except:
            return False