import pytest
from unittest.mock import AsyncMock

from app.core.models import ChatMessage, ChatRequest, Citation, Conversation
from app.core.tutor_models import EvidenceState, TeachingAction
from app.services.chat_service import STATE_UNCERTAINTY_MESSAGE, ChatService
from app.services.decision_controller import DecisionController
from app.services.evidence_assessor import EvidenceAssessor
from app.services.learner_state_stub import FixedLearnerStateEstimator
from app.services.prompt_builder import PromptBuilder


def cite(score, page=1, text="passage"):
    return Citation(document_id="doc1", document_name="notes.pdf", page_number=page, text=text, score=score)


@pytest.fixture
def mock_conversation_store():
    store = AsyncMock()
    store.create_conversation.return_value = Conversation(id="conv1", title="New Conv")
    store.get_conversation.return_value = None
    store.add_message = AsyncMock()
    return store


class MockLLMProvider:
    """Stub AI service: records every prompt, answers NO to conflict checks."""

    def __init__(self):
        self.calls = []

    async def generate_stream(self, messages, context):
        self.calls.append(messages)
        if "contradict" in messages[0].content:
            yield "NO"
        else:
            yield "Here is the answer."


class FixedEstimator:
    def __init__(self, state):
        self.state = state

    def estimate(self, conversation_id, topic=None):
        return self.state


def make_service(citations, store, llm, learner=None):
    retriever = AsyncMock()
    retriever.retrieve.return_value = citations
    return ChatService(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        llm=llm,
        conversation_store=store,
        evidence_assessor=EvidenceAssessor(llm=llm),
        learner_estimator=learner or FixedLearnerStateEstimator(),
        decision_controller=DecisionController(),
    )


async def run(service, message="What is a use case?"):
    events = [event async for event in service.chat_stream(ChatRequest(message=message))]
    text = "".join(chunk for chunk, _, _ in events)
    citations, meta = events[0][1], events[0][2]
    return text, citations, meta


@pytest.mark.asyncio
async def test_no_evidence_states_uncertainty_without_calling_llm(mock_conversation_store):
    llm = MockLLMProvider()
    text, citations, meta = await run(make_service([], mock_conversation_store, llm))

    assert text == STATE_UNCERTAINTY_MESSAGE
    assert citations == []
    assert meta["conversation_id"] == "conv1"
    assert meta["tutor_state"].evidence_state is EvidenceState.INSUFFICIENT
    assert meta["tutor_state"].teaching_action is TeachingAction.STATE_UNCERTAINTY
    assert llm.calls == []


@pytest.mark.asyncio
async def test_weak_evidence_is_insufficient(mock_conversation_store):
    llm = MockLLMProvider()
    text, citations, meta = await run(make_service([cite(0.45), cite(0.30)], mock_conversation_store, llm))

    assert text == STATE_UNCERTAINTY_MESSAGE
    assert meta["tutor_state"].evidence_score == 0.45
    assert llm.calls == []


@pytest.mark.asyncio
async def test_strong_evidence_uncertain_learner_gets_worked_example(mock_conversation_store):
    llm = MockLLMProvider()
    service = make_service([cite(0.82, page=3), cite(0.64, page=4), cite(0.20)], mock_conversation_store, llm)
    text, citations, meta = await run(service)

    state = meta["tutor_state"]
    assert text == "Here is the answer."
    assert state.learner_state == "uncertain"
    assert state.evidence_state is EvidenceState.STRONG
    assert state.teaching_action is TeachingAction.WORKED_EXAMPLE
    assert [c.score for c in citations] == [0.82, 0.64, 0.20]  # every passage reaches the LLM, best first

    answer_prompt = llm.calls[-1][0].content
    assert "Worked example" in answer_prompt
    assert "Check question:" in answer_prompt


def conversation_with(*user_questions):
    messages = []
    for q in user_questions:
        messages += [ChatMessage(role="user", content=q), ChatMessage(role="assistant", content="...")]
    return Conversation(id="conv1", title="Conv", messages=messages)


@pytest.mark.asyncio
async def test_follow_up_is_searched_with_previous_question(mock_conversation_store):
    mock_conversation_store.get_conversation.return_value = conversation_with(
        "tell me about external inputs", "explain again"
    )
    llm = MockLLMProvider()
    service = make_service([cite(0.60)], mock_conversation_store, llm)
    await run_in(service, "explain this topic again", "conv1")

    query = service._retriever.retrieve.call_args.kwargs["query"]
    assert query == "tell me about external inputs"


@pytest.mark.asyncio
async def test_new_question_is_not_treated_as_follow_up(mock_conversation_store):
    mock_conversation_store.get_conversation.return_value = conversation_with("tell me about external inputs")
    llm = MockLLMProvider()
    service = make_service([], mock_conversation_store, llm)
    text = await run_in(service, "What is the capital of France?", "conv1")

    assert service._retriever.retrieve.call_args.kwargs["query"] == "What is the capital of France?"
    assert text == STATE_UNCERTAINTY_MESSAGE


async def run_in(service, message, conversation_id):
    request = ChatRequest(message=message, conversation_id=conversation_id)
    return "".join([chunk async for chunk, _, _ in service.chat_stream(request)])


@pytest.mark.asyncio
async def test_confused_learner_gets_simplify(mock_conversation_store):
    llm = MockLLMProvider()
    service = make_service([cite(0.60)], mock_conversation_store, llm, learner=FixedEstimator("confused"))
    _, _, meta = await run(service)

    assert meta["tutor_state"].evidence_state is EvidenceState.PARTIAL
    assert meta["tutor_state"].teaching_action is TeachingAction.SIMPLIFY
