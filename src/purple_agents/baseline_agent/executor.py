"""
A2A Executor for Baseline Werewolf Agent.
"""
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    TaskState,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import new_agent_text_message, new_task

from .agent import BaselineWerewolfAgent

logger = logging.getLogger(__name__)


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected
}


class BaselineExecutor(AgentExecutor):
    """
    A2A Executor for Baseline Werewolf Agent.
    Handles incoming requests and routes them to agent instances.
    """
    
    def __init__(self, agent_id: str = "baseline-agent"):
        self.agent_id = agent_id
        self.agents: dict[str, BaselineWerewolfAgent] = {}
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute agent request.
        
        Args:
            context: Request context with message and task info
            event_queue: Queue for publishing events
        """
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message"))
        
        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(
                message=f"Task {task.id} already processed"
            ))
        
        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)
        
        # Get or create agent for this context
        context_id = task.context_id
        agent = self.agents.get(context_id)
        if not agent:
            agent = BaselineWerewolfAgent(agent_id=self.agent_id)
            self.agents[context_id] = agent
        
        updater = TaskUpdater(event_queue, task.id, context_id)
        await updater.start_work()
        
        try:
            await agent.run(msg, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            await updater.failed(new_agent_text_message(
                f"Agent error: {e}",
                context_id=context_id,
                task_id=task.id
            ))
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel operation not supported"""
        raise ServerError(error=UnsupportedOperationError())
