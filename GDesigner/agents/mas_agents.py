"""Quiet agents for the MASBench robustness experiments (no prompt printing, no wiki tools)."""
from typing import Any, Dict, List

from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.llm.llm_registry import LLMRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry


@AgentRegistry.register('MASAgent')
class MASAgent(Node):
    def __init__(self, id: str | None = None, role: str = None, domain: str = "", llm_name: str = ""):
        super().__init__(id, "MASAgent", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = role
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)

    def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        user_prompt = f"The task is: {raw_inputs['task']}\n"
        others = "".join(info['output'] for info in spatial_info.values())
        if others:
            user_prompt += f"\nThe outputs of other agents are as follows:\n{others}\n"
        return self.constraint, user_prompt

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        raise NotImplementedError("synchronous execution is not used")

    async def _async_execute(self, input: Dict[str, str], spatial_info: Dict[str, Dict], temporal_info: Dict[str, Dict], **kwargs):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        return await self.llm.agen(message)


@AgentRegistry.register('MASFinal')
class MASFinal(Node):
    def __init__(self, id: str | None = None, domain: str = "", llm_name: str = ""):
        super().__init__(id, "MASFinal", domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        system_prompt = f"{self.prompt_set.get_decision_role()}.\n {self.prompt_set.get_decision_constraint()}"
        others = "".join(info['output'] for info in spatial_info.values())
        user_prompt = f"The task is:\n\n {raw_inputs['task']}\n"
        if others:
            user_prompt += f"\nAt the same time, the output of other agents is as follows:\n{others}\n"
        return system_prompt, user_prompt

    def _execute(self, input, spatial_info, temporal_info, **kwargs):
        raise NotImplementedError("synchronous execution is not used")

    async def _async_execute(self, input, spatial_info, temporal_info, **kwargs):
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        return await self.llm.agen(message)
