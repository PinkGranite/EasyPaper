"""
MetaData Agent Router
- **Description**:
    - FastAPI router for MetaData-based paper generation
    - Independent API - can be called directly without frontend
    - Endpoints:
        - POST /metadata/generate - Generate complete paper (blocking)
        - POST /metadata/generate/stream - Generate with SSE progress streaming
        - POST /metadata/generate/{task_id}/feedback - Inject user feedback
        - POST /metadata/generate/section - Generate single section
        - GET /metadata/health - Health check
        - GET /metadata/schema - Get input schema
"""
import asyncio
import json
import uuid
from typing import Any, Dict, Optional, TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .models import (
    PaperGenerationRequest,
    PaperGenerationResult,
    PlanResult,
    SectionGenerationRequest,
    SectionResult,
)
from .progress import ProgressCallback

if TYPE_CHECKING:
    from .metadata_agent import MetaDataAgent


# In-memory task registry for active streaming generations
_active_tasks: Dict[str, Dict[str, Any]] = {}


def create_metadata_router(agent: "MetaDataAgent") -> APIRouter:
    """
    Create FastAPI router for MetaData Agent

    - **Args**:
        - `agent` (MetaDataAgent): MetaDataAgent instance

    - **Returns**:
        - `APIRouter`: FastAPI router with all endpoints
    """
    router = APIRouter(prefix="/metadata", tags=["MetaData Paper Generation"])

    # ------------------------------------------------------------------
    # POST /metadata/generate  (blocking, original behavior)
    # ------------------------------------------------------------------
    @router.post("/generate", response_model=PaperGenerationResult)
    async def generate_paper(request: PaperGenerationRequest) -> PaperGenerationResult:
        """Generate complete paper from MetaData (blocking call)."""
        try:
            metadata = request.to_metadata()
            result = await agent.generate_paper(
                metadata=metadata,
                output_dir=request.output_dir,
                save_output=request.save_output,
                compile_pdf=request.compile_pdf,
                template_path=request.template_path,
                figures_source_dir=request.figures_source_dir,
                target_pages=request.target_pages,
                enable_review=request.enable_review,
                max_review_iterations=request.max_review_iterations,
                enable_planning=request.enable_planning,
                enable_exemplar=request.enable_exemplar,
                enable_vlm_review=request.enable_vlm_review,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # POST /metadata/generate/stream  (SSE streaming)
    # ------------------------------------------------------------------
    @router.post("/generate/stream")
    async def generate_paper_stream(request: PaperGenerationRequest):
        """
        Generate paper with real-time SSE progress streaming.

        Returns a Server-Sent Events stream. Each event is a JSON object
        with a ``type`` field indicating the event kind.
        """
        task_id = str(uuid.uuid4())
        event_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        feedback_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        _active_tasks[task_id] = {
            "event_queue": event_queue,
            "feedback_queue": feedback_queue,
            "status": "running",
        }

        async def progress_callback(event: Dict[str, Any]) -> None:
            event["task_id"] = task_id
            await event_queue.put(event)

        async def run_generation() -> None:
            try:
                metadata = request.to_metadata()
                await agent.generate_paper(
                    metadata=metadata,
                    output_dir=request.output_dir,
                    save_output=request.save_output,
                    compile_pdf=request.compile_pdf,
                    template_path=request.template_path,
                    figures_source_dir=request.figures_source_dir,
                    target_pages=request.target_pages,
                    enable_review=request.enable_review,
                    max_review_iterations=request.max_review_iterations,
                    enable_planning=request.enable_planning,
                    enable_exemplar=request.enable_exemplar,
                    enable_vlm_review=request.enable_vlm_review,
                    enable_user_feedback=request.enable_user_feedback,
                    progress_callback=progress_callback,
                    feedback_queue=feedback_queue,
                    feedback_timeout=request.feedback_timeout
                    if hasattr(request, "feedback_timeout")
                    else 300.0,
                    artifacts_prefix=request.artifacts_prefix or "",
                )
            except Exception as e:
                await event_queue.put({
                    "type": "error",
                    "task_id": task_id,
                    "message": str(e),
                })
            finally:
                await event_queue.put(None)
                _active_tasks.pop(task_id, None)

        gen_task = asyncio.create_task(run_generation())
        _active_tasks[task_id]["task"] = gen_task

        async def event_stream():
            # First event: send task_id so the client can use it for feedback
            yield f"data: {json.dumps({'type': 'task_created', 'task_id': task_id})}\n\n"
            try:
                while True:
                    event = await event_queue.get()
                    if event is None:
                        break
                    yield f"data: {json.dumps(event, default=str)}\n\n"
            except asyncio.CancelledError:
                gen_task.cancel()
            finally:
                if not gen_task.done():
                    gen_task.cancel()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # POST /metadata/prepare-plan  (blocking, returns PlanResult JSON)
    # ------------------------------------------------------------------
    @router.post("/prepare-plan", response_model=PlanResult)
    async def prepare_plan(request: PaperGenerationRequest) -> PlanResult:
        """
        Run planning phases only and return a serializable PlanResult.

        - **Description**:
            - Executes Phases 0 through 0.5 (planning, ref discovery, DAG,
              code context, table conversion) without starting generation.
            - The returned PlanResult can be reviewed/modified by the user,
              then sent to ``/metadata/generate-from-plan/stream``.
        """
        try:
            metadata = request.to_metadata()
            result = await agent.prepare_plan(
                metadata=metadata,
                template_path=request.template_path,
                target_pages=request.target_pages,
                enable_planning=request.enable_planning,
                enable_exemplar=request.enable_exemplar,
                save_output=request.save_output,
                output_dir=request.output_dir,
                artifacts_prefix=request.artifacts_prefix or "",
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # POST /metadata/generate-from-plan/stream  (SSE, resumes from PlanResult)
    # ------------------------------------------------------------------
    @router.post("/generate-from-plan/stream")
    async def generate_from_plan_stream(plan_result: PlanResult):
        """
        Generate paper content from a (possibly user-modified) PlanResult.

        - **Description**:
            - Accepts the output of ``/metadata/prepare-plan``, reconstructs
              internal state, then runs Phases 1-5 with SSE streaming.
        """
        task_id = str(uuid.uuid4())
        event_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        feedback_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        _active_tasks[task_id] = {
            "event_queue": event_queue,
            "feedback_queue": feedback_queue,
            "status": "running",
        }

        async def progress_callback(event: Dict[str, Any]) -> None:
            event["task_id"] = task_id
            await event_queue.put(event)

        async def run_generation() -> None:
            try:
                await agent.execute_generation(
                    plan_result=plan_result,
                    enable_review=True,
                    max_review_iterations=3,
                    compile_pdf=True,
                    progress_callback=progress_callback,
                    feedback_queue=feedback_queue,
                )
            except Exception as e:
                await event_queue.put({
                    "type": "error",
                    "task_id": task_id,
                    "message": str(e),
                })
            finally:
                await event_queue.put(None)
                _active_tasks.pop(task_id, None)

        gen_task = asyncio.create_task(run_generation())
        _active_tasks[task_id]["task"] = gen_task

        async def event_stream():
            yield f"data: {json.dumps({'type': 'task_created', 'task_id': task_id})}\n\n"
            try:
                while True:
                    event = await event_queue.get()
                    if event is None:
                        break
                    yield f"data: {json.dumps(event, default=str)}\n\n"
            except asyncio.CancelledError:
                gen_task.cancel()
            finally:
                if not gen_task.done():
                    gen_task.cancel()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # POST /metadata/generate/{task_id}/feedback
    # ------------------------------------------------------------------
    @router.post("/generate/{task_id}/feedback")
    async def submit_feedback(task_id: str, feedback: Dict[str, Any]):
        """
        Submit user feedback during review loop.

        - **Args**:
            - `task_id` (str): Task ID from the ``task_created`` SSE event.
            - `feedback` (dict): Feedback payload with fields:
                - ``feedback_text`` (str): User's review comments
                - ``section_targets`` (list[str], optional): Target sections
                - ``action`` (str): One of ``continue``, ``accept``, ``stop``
        """
        task_info = _active_tasks.get(task_id)
        if not task_info:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found or already completed",
            )
        await task_info["feedback_queue"].put(feedback)
        return {"status": "ok", "task_id": task_id}

    # ------------------------------------------------------------------
    # POST /metadata/generate/{task_id}/cancel
    # ------------------------------------------------------------------
    @router.post("/generate/{task_id}/cancel")
    async def cancel_generation(task_id: str):
        """
        Cancel a running generation task.
        """
        task_info = _active_tasks.get(task_id)
        if not task_info:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found or already completed",
            )
        gen_task = task_info.get("task")
        if gen_task and not gen_task.done():
            gen_task.cancel()
            task_info["status"] = "cancelled"
            return {"status": "ok", "task_id": task_id}
        return {"status": "ok", "message": "Task already completed"}

    # ------------------------------------------------------------------
    # POST /metadata/generate/{task_id}/resume
    # ------------------------------------------------------------------
    @router.post("/generate/{task_id}/resume")
    async def resume_generation(task_id: str, feedback: Dict[str, Any]):
        """
        Resume generation from checkpoint after user feedback.

        - **Description**:
            - Loads the checkpoint saved during the feedback pause, injects user
              annotations, and returns a new SSE stream for the remaining work.

        - **Args**:
            - `task_id` (str): Original task ID.
            - `feedback` (dict): Enhanced feedback payload with action,
              global_feedback, section_annotations.
        """
        checkpoint_path = feedback.get("checkpoint_path", "")
        if not checkpoint_path:
            raise HTTPException(status_code=400, detail="checkpoint_path is required")

        import os
        if not os.path.isfile(checkpoint_path):
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")

        new_task_id = task_id
        event_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

        _active_tasks[new_task_id] = {
            "event_queue": event_queue,
            "status": "running",
        }

        async def progress_callback(event: Dict[str, Any]) -> None:
            event["task_id"] = new_task_id
            await event_queue.put(event)

        async def run_resumed() -> None:
            try:
                await agent.resume_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    user_feedback=feedback,
                    progress_callback=progress_callback,
                    artifacts_prefix=feedback.get("artifacts_prefix", ""),
                )
            except Exception as e:
                await event_queue.put({
                    "type": "error",
                    "task_id": new_task_id,
                    "message": str(e),
                })
            finally:
                await event_queue.put(None)
                _active_tasks.pop(new_task_id, None)

        gen_task = asyncio.create_task(run_resumed())
        _active_tasks[new_task_id]["task"] = gen_task

        async def event_stream():
            yield f"data: {json.dumps({'type': 'task_resumed', 'task_id': new_task_id})}\n\n"
            try:
                while True:
                    event = await event_queue.get()
                    if event is None:
                        break
                    yield f"data: {json.dumps(event, default=str)}\n\n"
            except asyncio.CancelledError:
                gen_task.cancel()
            finally:
                if not gen_task.done():
                    gen_task.cancel()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # POST /metadata/generate/section  (single section, unchanged)
    # ------------------------------------------------------------------
    @router.post("/generate/section", response_model=SectionResult)
    async def generate_single_section(
        request: SectionGenerationRequest,
    ) -> SectionResult:
        """Generate a single section (for debugging or incremental generation)."""
        try:
            result = await agent.generate_single_section(request)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # GET /metadata/health
    # ------------------------------------------------------------------
    @router.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "ok",
            "agent": "metadata_agent",
            "model": agent.model_name,
            "description": "MetaData-based paper generation (Simple Mode)",
            "active_tasks": len(_active_tasks),
            "endpoints": [
                "POST /metadata/generate - Generate complete paper",
                "POST /metadata/generate/stream - Generate with SSE streaming",
                "POST /metadata/generate/{task_id}/feedback - Submit review feedback",
                "POST /metadata/generate/section - Generate single section",
                "GET /metadata/health - Health check",
                "GET /metadata/schema - Get input schema",
            ],
        }

    # ------------------------------------------------------------------
    # GET /metadata/schema
    # ------------------------------------------------------------------
    @router.get("/schema")
    async def get_input_schema():
        """Get the input schema for paper generation."""
        return {
            "input_schema": {
                "title": {
                    "type": "string",
                    "description": "Paper title",
                    "required": False,
                    "default": "Untitled Paper",
                },
                "idea_hypothesis": {
                    "type": "string",
                    "description": "Research idea or hypothesis (natural language)",
                    "required": True,
                },
                "method": {
                    "type": "string",
                    "description": "Method/approach description (natural language)",
                    "required": True,
                },
                "data": {
                    "type": "string",
                    "description": "Data or validation method description",
                    "required": True,
                },
                "experiments": {
                    "type": "string",
                    "description": "Experiment design, execution, results, findings",
                    "required": True,
                },
                "references": {
                    "type": "array",
                    "items": "string (BibTeX entry)",
                    "description": "List of BibTeX reference entries",
                    "required": False,
                    "default": [],
                },
                "template_path": {
                    "type": "string",
                    "description": "Path to .zip template file for PDF compilation",
                    "required": False,
                },
                "style_guide": {
                    "type": "string",
                    "description": "Writing style guide (e.g., 'ICML', 'NeurIPS')",
                    "required": False,
                },
                "compile_pdf": {
                    "type": "boolean",
                    "description": "Whether to compile PDF (requires template_path)",
                    "required": False,
                    "default": True,
                },
                "save_output": {
                    "type": "boolean",
                    "description": "Whether to save output files to disk",
                    "required": False,
                    "default": True,
                },
            },
            "example": {
                "title": "TransKG: Knowledge Graph Completion with Transformers",
                "idea_hypothesis": "We hypothesize that pre-trained Transformer models can better capture semantic relationships...",
                "method": "We propose TransKG, combining BERT with relation-aware attention...",
                "data": "We evaluate on FB15k-237, WN18RR, and YAGO3-10 datasets...",
                "experiments": "Compared against TransE, RotatE, achieving 0.391 MRR...",
                "references": [
                    "@inproceedings{bordes2013transE, title={Translating embeddings}, author={Bordes}, year={2013}}"
                ],
                "style_guide": "ICML",
                "compile_pdf": True,
            },
        }

    # ------------------------------------------------------------------
    # POST /metadata/generate-from-folder
    # ------------------------------------------------------------------

    @router.post("/generate-from-folder")
    async def generate_metadata_from_folder(
        folder_path: str,
        title: Optional[str] = None,
        style_guide: Optional[str] = None,
        template_path: Optional[str] = None,
        target_pages: Optional[int] = None,
    ):
        """
        Scan a folder of research materials and generate PaperMetaData.

        - **Args**:
            - `folder_path` (str): Path to a local folder containing
              research materials (PDFs, code, notes, bib, data, images).
            - `title` (str, optional): Override for the paper title.
            - `style_guide` (str, optional): Target venue (e.g. ICML).
            - `template_path` (str, optional): Path to LaTeX template.
            - `target_pages` (int, optional): Target page count.

        - **Returns**:
            - ``PaperMetaData`` as JSON.
        """
        from .metadata_generator import generate_metadata_from_folder as _gen

        overrides = {}
        if title:
            overrides["title"] = title
        if style_guide:
            overrides["style_guide"] = style_guide
        if template_path:
            overrides["template_path"] = template_path
        if target_pages:
            overrides["target_pages"] = target_pages

        try:
            result = await _gen(
                folder_path=folder_path,
                llm_client=agent.client,
                model_name=agent.model_name,
                **overrides,
            )
            return result.model_dump()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Metadata generation failed: {exc}")

    # ------------------------------------------------------------------
    # Docling — standalone PDF parsing endpoints
    # ------------------------------------------------------------------

    @router.post("/docling/parse")
    async def docling_parse(pdf_path: str):
        """
        Parse a local PDF file into structured academic sections.
        - **Args**:
            - `pdf_path` (str): Absolute path to a PDF on the server.
        - **Returns**:
            - ``DoclingPaperResult`` as JSON.
        """
        from pathlib import Path as _Path
        from ..shared.docling_service import DoclingService

        p = _Path(pdf_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_path}")

        docling_cfg = None
        if agent.tools_config and getattr(agent.tools_config, "docling", None):
            docling_cfg = agent.tools_config.docling

        svc = DoclingService(config=docling_cfg)
        try:
            result = svc.parse_pdf(p)
        except ImportError as exc:
            raise HTTPException(status_code=501, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Parse failed: {exc}")

        return DoclingService.result_to_dict(result)

    @router.post("/docling/download-and-parse")
    async def docling_download_and_parse(
        url: str,
        dest_dir: Optional[str] = None,
    ):
        """
        Download a PDF from URL and parse it into structured sections.
        - **Args**:
            - `url` (str): Direct PDF URL (e.g. arXiv PDF link).
            - `dest_dir` (str, optional): Save PDF to this directory.
        - **Returns**:
            - ``DoclingPaperResult`` as JSON.
        """
        from ..shared.docling_service import DoclingService

        docling_cfg = None
        if agent.tools_config and getattr(agent.tools_config, "docling", None):
            docling_cfg = agent.tools_config.docling

        svc = DoclingService(config=docling_cfg)
        try:
            result = await svc.download_and_parse(
                url, dest_dir=dest_dir, cleanup=(dest_dir is None),
            )
        except ImportError as exc:
            raise HTTPException(status_code=501, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Download/parse failed: {exc}")

        if not result.full_text:
            raise HTTPException(status_code=422, detail="PDF downloaded but parsing yielded no content")

        return DoclingService.result_to_dict(result)

    @router.post("/docling/enrich-refs")
    async def docling_enrich_refs(refs: list):
        """
        Batch-enrich reference dicts with Docling full-text analysis.
        - **Args**:
            - `refs` (list): Reference dicts with optional
              ``open_access_pdf`` / ``arxiv_id`` fields.
        - **Returns**:
            - Enriched refs list with ``docling_sections`` and ``docling_full_text``.
        """
        from ..shared.docling_service import DoclingService

        docling_cfg = None
        if agent.tools_config and getattr(agent.tools_config, "docling", None):
            docling_cfg = agent.tools_config.docling

        svc = DoclingService(config=docling_cfg)
        try:
            enriched = await svc.enrich_refs(refs)
        except ImportError as exc:
            raise HTTPException(status_code=501, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Enrichment failed: {exc}")

        return enriched

    return router
