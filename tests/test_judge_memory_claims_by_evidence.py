import pytest
from judge_memory import JudgeMemoryService, JudgeMemoryConfig


@pytest.mark.asyncio
async def test_claims_can_be_retrieved_by_evidence(tmp_path):
    service = JudgeMemoryService(
        JudgeMemoryConfig(
            data_dir=str(tmp_path),
            enable_fluid_memory=False,
        )
    )
    evidence = await service.ingest_evidence(
        raw_text="The court issued an order.",
        source_type="court_record",
    )
    claim = await service.add_claim(
        evidence_id=evidence.evidence_id,
        claim_text="The court issued an order.",
        claim_type="fact",
    )
    claims = service.claims_manager.get_claims_for_evidence(evidence.evidence_id)
    assert len(claims) == 1
    assert claims[0].claim_id == claim.claim_id
    assert claims[0].evidence_id == evidence.evidence_id
