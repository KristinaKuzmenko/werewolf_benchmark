"""
A2A AgentExecutor implementation for Werewolf Green Agent.
"""
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Task,
    TaskState,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    new_agent_text_message,
    new_task,
)

from .agent import WerewolfAgent


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected
}


class WerewolfExecutor(AgentExecutor):
    """
    Executor for handling Werewolf game orchestration requests.
    Implements A2A AgentExecutor interface.
    """
    
    def __init__(self):
        self.agents: dict[str, WerewolfAgent] = {}  # context_id -> agent instance

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute a Werewolf game orchestration request.
        
        Args:
            context: Request context with message and task info
            event_queue: Queue for publishing events (status updates, results)
        """
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(
                message=f"Task {task.id} already processed (state: {task.status.state})"
            ))

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self.agents.get(context_id)
        if not agent:
            agent = WerewolfAgent()
            self.agents[context_id] = agent

        updater = TaskUpdater(event_queue, task.id, context_id)

        await updater.start_work()
        try:
            await agent.run(msg, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as e:
            print(f"Task failed with agent error: {e}")
            await updater.failed(new_agent_text_message(
                f"Agent error: {e}", 
                context_id=context_id, 
                task_id=task.id
            ))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel operation not supported for Werewolf games."""
        raise ServerError(error=UnsupportedOperationError())
