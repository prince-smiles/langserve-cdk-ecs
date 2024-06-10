from langchain.agents import AgentExecutor
from langchain_core.tools import BaseTool

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from langchain_core.tools import BaseTool
from langchain_core.agents import AgentAction, AgentFinish, AgentStep

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    BaseCallbackManager,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)
from langchain.agents.tools import InvalidTool


async def _aperform_agent_action(
    self,
    name_to_tool_map: Dict[str, BaseTool],
    color_mapping: Dict[str, str],
    agent_action: AgentAction,
    run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
) -> AgentStep:
    if run_manager:
        await run_manager.on_agent_action(
            agent_action, verbose=self.verbose, color="green"
        )
    # Otherwise we lookup the tool
    # print(run_manager)
    print(run_manager.metadata["session_id"])
    if agent_action.tool in name_to_tool_map:
        tool = name_to_tool_map[agent_action.tool]
        return_direct = tool.return_direct
        color = color_mapping[agent_action.tool]
        tool_run_kwargs = self.agent.tool_run_logging_kwargs()
        if return_direct:
            tool_run_kwargs["llm_prefix"] = ""
        # We then call the tool on the tool input to get an observation
        agent_action.tool_input["session_id"] = run_manager.metadata["session_id"]
        print("inside aperform agent action", agent_action.tool_input)
        observation = await tool.arun(
            agent_action.tool_input,
            verbose=self.verbose,
            color=color,
            callbacks=run_manager.get_child() if run_manager else None,
            **tool_run_kwargs,
        )
        print("observation", observation)
    else:
        tool_run_kwargs = self.agent.tool_run_logging_kwargs()
        observation = await InvalidTool().arun(
            {
                "requested_tool_name": agent_action.tool,
                "available_tool_names": list(name_to_tool_map.keys()),
            },
            verbose=self.verbose,
            color=None,
            callbacks=run_manager.get_child() if run_manager else None,
            **tool_run_kwargs,
        )
    return AgentStep(action=agent_action, observation=observation)




AgentExecutor._aperform_agent_action = _aperform_agent_action



import uuid
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableSerializable,
    ensure_config,
)

from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForToolRun,
    BaseCallbackManager,
    CallbackManager,
    CallbackManagerForToolRun,
)
from inspect import signature


from langchain_core.runnables.config import (
    _set_config_context,
    patch_config,
    run_in_executor,
)
from contextvars import copy_context
from langchain_core.runnables.utils import accepts_context
import asyncio

from langchain_core.pydantic_v1 import (
    BaseModel,
    Extra,
    Field,
    ValidationError,
    create_model,
    root_validator,
    validate_arguments,
)

class ToolException(Exception):
    """Optional exception that tool throws when execution error occurs.

    When this exception is thrown, the agent will not stop working,
    but it will handle the exception according to the handle_tool_error
    variable of the tool, and the processing result will be returned
    to the agent as observation, and printed in red on the console.
    """

    pass

async def arun(
    self,
    tool_input: Union[str, Dict],
    verbose: Optional[bool] = None,
    start_color: Optional[str] = "green",
    color: Optional[str] = "green",
    callbacks: Callbacks = None,
    *,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
    run_id: Optional[uuid.UUID] = None,
    config: Optional[RunnableConfig] = None,
    **kwargs: Any,
) -> Any:
    """Run the tool asynchronously."""
    if not self.verbose and verbose is not None:
        verbose_ = verbose
    else:
        verbose_ = self.verbose
    callback_manager = AsyncCallbackManager.configure(
        callbacks,
        self.callbacks,
        verbose_,
        tags,
        self.tags,
        metadata,
        self.metadata,
    )
    new_arg_supported = signature(self._arun).parameters.get("run_manager")
    run_manager = await callback_manager.on_tool_start(
        {"name": self.name, "description": self.description},
        tool_input if isinstance(tool_input, str) else str(tool_input),
        color=start_color,
        name=run_name,
        inputs=tool_input,
        run_id=run_id,
        **kwargs,
    )
    try:
        print("tool_input", tool_input)
        parsed_input = self._parse_input(tool_input)
        if ("session_id" in tool_input):
            parsed_input["session_id"] = tool_input["session_id"]
        # We then call the tool on the tool input to get an observation
        tool_args, tool_kwargs = self._to_args_and_kwargs(parsed_input)
        child_config = patch_config(
            config,
            callbacks=run_manager.get_child(),
        )
        context = copy_context()
        context.run(_set_config_context, child_config)
        coro = (
            context.run(
                self._arun, *tool_args, run_manager=run_manager, **tool_kwargs
            )
            if new_arg_supported
            else context.run(self._arun, *tool_args, **tool_kwargs)
        )
        if accepts_context(asyncio.create_task):
            observation = await asyncio.create_task(coro, context=context)  # type: ignore
        else:
            observation = await coro

    except ValidationError as e:
        if not self.handle_validation_error:
            raise e
        elif isinstance(self.handle_validation_error, bool):
            observation = "Tool input validation error"
        elif isinstance(self.handle_validation_error, str):
            observation = self.handle_validation_error
        elif callable(self.handle_validation_error):
            observation = self.handle_validation_error(e)
        else:
            raise ValueError(
                f"Got unexpected type of `handle_validation_error`. Expected bool, "
                f"str or callable. Received: {self.handle_validation_error}"
            )
        return observation
    except ToolException as e:
        if not self.handle_tool_error:
            await run_manager.on_tool_error(e)
            raise e
        elif isinstance(self.handle_tool_error, bool):
            if e.args:
                observation = e.args[0]
            else:
                observation = "Tool execution error"
        elif isinstance(self.handle_tool_error, str):
            observation = self.handle_tool_error
        elif callable(self.handle_tool_error):
            observation = self.handle_tool_error(e)
        else:
            raise ValueError(
                f"Got unexpected type of `handle_tool_error`. Expected bool, str "
                f"or callable. Received: {self.handle_tool_error}"
            )
        await run_manager.on_tool_end(
            observation, color="red", name=self.name, **kwargs
        )
        return observation
    except (Exception, KeyboardInterrupt) as e:
        await run_manager.on_tool_error(e)
        raise e
    else:
        await run_manager.on_tool_end(
            observation, color=color, name=self.name, **kwargs
        )
        return observation


BaseTool.arun = arun