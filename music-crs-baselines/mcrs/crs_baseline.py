import os
import torch
from typing import Optional, Any, List, Dict
from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.retrieval_modules import load_retrieval_module

class CRS_BASELINE:
    """
    Conversational Recommender System (CRS) baseline that wires together an LLM module and an item retrieval module over a music catalog and user profiles.
    Attributes:
        cache_dir: Local path for caching artifacts and indices.
        lm_type: Identifier/name for the LLM backend to load.
        retrieval_type: Retrieval backend to use (e.g., "bm25").
        item_db_name: Hugging Face dataset or DB name for item metadata.
        user_db_name: Hugging Face dataset or DB name for user metadata.
        split_types: Dataset split names to load (e.g., ["test_warm", "test_cold"]).
        corpus_types: Item fields used for retrieval (e.g., title, artist, album).
        device: Compute device for the LLM (e.g., "cuda", "cpu").
        dtype: Torch dtype used by the LLM.
        lm: Loaded LLM module used for response generation.
        retrieval: Retrieval module used to fetch candidate items.
        item_db: Item metadata database accessor.
        user_db: User profile database accessor.
        prompts_dir: Directory containing prompt templates.
        role_prompt: Loaded prompt templates keyed by role.
        session_memory: In-memory list of message dicts for the current session.
    """
    def __init__(self,
        lm_type="meta-llama/Llama-3.2-1B-Instruct",
        retrieval_type="bm25",
        item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
        track_split_types: list[str] = ["all_tracks"], # for test
        user_split_types: list[str] = ["all_users"],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir="./cache",
        device="cuda",
        attn_implementation="eager",
        dtype=torch.bfloat16,
        reranker_type=None,
        reranker_kwargs=None,
        retrieve_topk=20,
        final_topk=20,
        retrieval_kwargs=None,
        include_user_profile_in_retrieval=False,
        response_topk=1,
        generation_mode: str = "default",
        max_response_tokens: int = 64,
        lm_kwargs=None,
        load_lm: bool = True,
    ):
        """Initialize the CRS baseline components.

        Args:
            lm_type: LLM model identifier to load for response generation.
            retrieval_type: Retrieval backend name (e.g., "bm25").
            item_db_name: Dataset/DB name for item metadata.
            user_db_name: Dataset/DB name for user metadata.
            split_types: Dataset split names to load.
            corpus_types: Item metadata fields used for retrieval.
            cache_dir: Local directory for caching artifacts/indices.
            device: Compute device for the LLM (e.g., "cuda", "cpu").
            dtype: Torch dtype for the LLM weights/tensors.
        """
        self.cache_dir = cache_dir
        self.lm_type = lm_type
        self.retrieval_type = retrieval_type
        self.item_db_name = item_db_name
        self.user_db_name = user_db_name
        self.track_split_types = track_split_types
        self.user_split_types = user_split_types
        self.corpus_types = corpus_types
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.retrieve_topk = retrieve_topk
        self.final_topk = final_topk
        self.include_user_profile_in_retrieval = include_user_profile_in_retrieval
        self.response_topk = max(1, response_topk)
        self.generation_mode = generation_mode
        self.max_response_tokens = max_response_tokens
        self.lm = None
        if load_lm:
            from mcrs.lm_modules import load_lm_module

            self.lm = load_lm_module(
                self.lm_type,
                self.device,
                self.attn_implementation,
                self.dtype,
                lm_kwargs=lm_kwargs or {},
            )
        self.retrieval = load_retrieval_module(
            self.retrieval_type,
            self.item_db_name,
            self.track_split_types,
            self.corpus_types,
            self.cache_dir,
            device=self.device,
            retrieval_kwargs=retrieval_kwargs or {},
        )
        self.reranker = None
        if reranker_type:
            from mcrs.reranker_modules import load_reranker_module

            reranker_kwargs = reranker_kwargs or {}
            reranker_kwargs.setdefault("device", self.device)
            self.reranker = load_reranker_module(reranker_type, cache_dir=self.cache_dir, **reranker_kwargs)
        self.item_db = MusicCatalogDB(self.item_db_name, self.track_split_types, self.corpus_types)
        self.user_db = UserProfileDB(self.user_db_name, self.user_split_types)
        self.prompts_dir = os.path.join(os.path.dirname(__file__), "system_prompts")
        self.role_prompt = {
            "role_play": open(f"{self.prompts_dir}/roleplay.txt", "r", encoding="utf-8").read(),
            "personalization": open(f"{self.prompts_dir}/personalization.txt", "r", encoding="utf-8").read(),
            "response_generation": self._load_response_prompt(),
        }
        self.session_memory = []

    def _load_response_prompt(self) -> str:
        if self.generation_mode == "gpt_curator":
            path = f"{self.prompts_dir}/response_generation_gpt_curator.txt"
        elif self.generation_mode == "grounded":
            path = f"{self.prompts_dir}/response_generation_grounded.txt"
        else:
            path = f"{self.prompts_dir}/response_generation.txt"
        return open(path, "r", encoding="utf-8").read()

    def _postprocess_retrieval(
        self,
        candidates: List[str],
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        dialogue_query: Optional[str] = None,
        session_memory: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        if self.reranker is not None:
            kwargs = {
                "user_id": user_id,
                "topk": self.final_topk,
                "query": query,
                "dialogue_query": dialogue_query,
                "item_db": self.item_db,
                "retrieval": self.retrieval,
                "user_db": self.user_db,
                "session_memory": session_memory,
            }
            candidates = self.reranker.rerank(candidates, **kwargs)
        if len(candidates) > self.final_topk:
            candidates = candidates[: self.final_topk]
        return candidates

    def _batch_postprocess_retrieval(
        self,
        batch_candidates: List[List[str]],
        user_ids: List[Optional[str]],
        queries: Optional[List[str]] = None,
        dialogue_queries: Optional[List[str]] = None,
        session_memories: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> List[List[str]]:
        if self.reranker is not None and hasattr(self.reranker, "batch_rerank"):
            kwargs = {"topk": self.final_topk}
            if queries is not None:
                kwargs["queries"] = queries
            if dialogue_queries is not None and getattr(self.reranker, "needs_retrieval", False):
                kwargs["dialogue_queries"] = dialogue_queries
            if getattr(self.reranker, "needs_item_db", False):
                kwargs["item_db"] = self.item_db
            if getattr(self.reranker, "needs_retrieval", False):
                kwargs["retrieval"] = self.retrieval
            kwargs["user_db"] = self.user_db
            if session_memories is not None:
                kwargs["session_memories"] = session_memories
            reranked = self.reranker.batch_rerank(batch_candidates, user_ids, **kwargs)
            return [items[: self.final_topk] for items in reranked]
        return [
            self._postprocess_retrieval(
                candidates,
                user_id,
                query=query,
                dialogue_query=dialogue_query,
                session_memory=session_memory,
            )
            for candidates, user_id, query, dialogue_query, session_memory in zip(
                batch_candidates,
                user_ids,
                queries or [None] * len(batch_candidates),
                dialogue_queries or queries or [None] * len(batch_candidates),
                session_memories or [None] * len(batch_candidates),
            )
        ]

    def _reset_session_memory(self):
        """Clear all messages stored in the current session memory.
        """
        self.session_memory = []

    def _upload_session_memory(self, chat_history: List[Dict[str, Any]]):
        """Upload the session memory to the database.
        """
        self.session_memory = chat_history

    def _get_system_prompt(self, user_id: Optional[str] = None) -> str:
        """Build the system prompt, optionally personalized with a user profile.
        Args:
            user_id: Optional user identifier. When provided, includes a personalization segment derived from the user's profile.
        Returns:
            The final system prompt string used for the LLM.
        """
        system_prompt = self.role_prompt["role_play"] + self.role_prompt["response_generation"]
        if user_id:
            user_profile_str = self.user_db.id_to_profile_str(user_id)
            system_prompt += self.role_prompt["personalization"] + '\n' + user_profile_str
        return system_prompt

    def _build_retrieval_input(self, session_memory: List[Dict[str, Any]], user_id: Optional[str] = None) -> str:
        lines = []
        if user_id and self.include_user_profile_in_retrieval:
            lines.append(f"user_profile: {self.user_db.id_to_profile_str(user_id)}")
        lines.extend(
            f"{conversation['role']}: {conversation['content']}" for conversation in session_memory
        )
        return "\n".join(lines)

    def _build_dialogue_input(self, session_memory: List[Dict[str, Any]]) -> str:
        return "\n".join(
            f"{conversation['role']}: {conversation['content']}" for conversation in session_memory
        )

    def _conversation_summary(self, session_memory: List[Dict[str, Any]]) -> str:
        user_turns = [turn["content"] for turn in session_memory if turn.get("role") == "user"]
        if not user_turns:
            return "No prior user messages."
        if len(user_turns) == 1:
            return f"The user wants: {user_turns[0]}"
        earlier = " | ".join(user_turns[:-1][-2:])
        return f"Earlier preferences: {earlier}. Latest request: {user_turns[-1]}"

    def _build_generation_context(
        self,
        track_ids: List[str],
        user_query: str,
        session_memory: List[Dict[str, Any]],
    ) -> str:
        if self.generation_mode != "grounded":
            return self._format_recommend_items(track_ids)

        lines = [
            "=== RETRIEVAL RESULTS (use only these tracks) ===",
            f"Conversation summary: {self._conversation_summary(session_memory)}",
            f"User's latest request: {user_query}",
            "Ranked recommendations:",
        ]
        for index, track_id in enumerate(track_ids[: self.response_topk], start=1):
            lines.append(f"  #{index}: {self.item_db.id_to_metadata(track_id)}")
        lines.append("=== END RETRIEVAL RESULTS ===")
        return "\n".join(lines)

    def _format_recommend_items(self, track_ids: List[str]) -> str:
        lines = []
        for index, track_id in enumerate(track_ids[: self.response_topk], start=1):
            lines.append(f"{index}. {self.item_db.id_to_metadata(track_id)}")
        return "\n".join(lines)

    def chat(self, user_query: str, user_id: Optional[str] = None) -> dict[str, Any]:
        """Run a single CRS turn: retrieve items and generate a response.
        Args:
            user_query: The user's latest message or request.
            user_id: Optional user identifier for personalization.
        Returns:
            A dictionary with keys:
                - user_id: The user identifier (may be None).
                - user_query: Echo of the input query.
                - retrieval_items: List of retrieved item IDs (top candidates).
                - recommend_item: Metadata for the top recommended item.
                - response: The generated assistant response string.
        """
        self.session_memory.append({"role": "user", "content": user_query})
        # stage0. system prompt
        system_prompt = self._get_system_prompt(user_id)
        # stage1. retrieval
        retrieval_input = self._build_retrieval_input(self.session_memory, user_id=user_id)
        retrieval_items = self.retrieval.text_to_item_retrieval(retrieval_input, topk=self.retrieve_topk)
        retrieval_items = self._postprocess_retrieval(
            retrieval_items,
            user_id=user_id,
            query=retrieval_input,
            dialogue_query=self._build_dialogue_input(self.session_memory),
            session_memory=self.session_memory,
        )
        recommend_item = self._build_generation_context(
            retrieval_items, user_query, self.session_memory
        )
        response = self.lm.response_generation(
            system_prompt,
            self.session_memory,
            recommend_item,
            max_new_tokens=self.max_response_tokens,
            generation_mode=self.generation_mode,
        )
        return {
            "user_id": user_id,
            "user_query": user_query,
            "retrieval_items": retrieval_items,
            "recommend_item": recommend_item,
            "response": response,
        }

    def batch_chat(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run multiple CRS turns in batch: retrieve items and generate responses.
        Args:
            batch_data: List of dictionaries, each containing:
                - user_query: The user's latest message or request.
                - user_id: Optional user identifier for personalization.
                - session_memory: List of chat history messages.
        Returns:
            A list of dictionaries, each with keys:
                - user_id: The user identifier (may be None).
                - user_query: Echo of the input query.
                - retrieval_items: List of retrieved item IDs (top candidates).
                - recommend_item: Metadata for the top recommended item.
                - response: The generated assistant response string.
        """
        # Prepare batch inputs
        sys_prompts = []
        retrieval_inputs = []
        dialogue_inputs = []
        session_memories = []

        for data in batch_data:
            user_query = data['user_query']
            user_id = data.get('user_id')
            session_memory = data['session_memory'].copy()
            session_memory.append({"role": "user", "content": user_query})

            sys_prompts.append(self._get_system_prompt(user_id))
            retrieval_input = self._build_retrieval_input(session_memory, user_id=user_id)
            retrieval_inputs.append(retrieval_input)
            dialogue_inputs.append(self._build_dialogue_input(session_memory))
            session_memories.append(session_memory)

        # Stage 1: Batch retrieval
        if hasattr(self.retrieval, "batch_retrieve"):
            user_ids = [data.get("user_id") for data in batch_data]
            batch_retrieval_items = self.retrieval.batch_retrieve(
                full_queries=retrieval_inputs,
                dialogue_queries=dialogue_inputs,
                user_ids=user_ids,
                topk=self.retrieve_topk,
            )
        elif hasattr(self.retrieval, 'batch_text_to_item_retrieval'):
            batch_retrieval_items = self.retrieval.batch_text_to_item_retrieval(retrieval_inputs, topk=self.retrieve_topk)
        else:
            # Fallback to sequential retrieval if batch method not available
            batch_retrieval_items = [self.retrieval.text_to_item_retrieval(inp, topk=self.retrieve_topk) for inp in retrieval_inputs]

        user_ids = [data.get('user_id') for data in batch_data]
        batch_retrieval_items = self._batch_postprocess_retrieval(
            batch_retrieval_items,
            user_ids,
            queries=retrieval_inputs,
            dialogue_queries=dialogue_inputs,
            session_memories=session_memories,
        )
        recommend_items = [
            self._build_generation_context(items, data["user_query"], session_memories[i])
            for i, (items, data) in enumerate(zip(batch_retrieval_items, batch_data))
        ]

        # Stage 2: Batch response generation
        if hasattr(self.lm, 'batch_response_generation'):
            responses = self.lm.batch_response_generation(
                sys_prompts,
                session_memories,
                recommend_items,
                max_new_tokens=self.max_response_tokens,
                generation_mode=self.generation_mode,
            )
        else:
            responses = [
                self.lm.response_generation(
                    sys_prompts[i],
                    session_memories[i],
                    recommend_items[i],
                    max_new_tokens=self.max_response_tokens,
                    generation_mode=self.generation_mode,
                )
                for i in range(len(batch_data))
            ]

        # Prepare results
        results = []
        for i, data in enumerate(batch_data):
            results.append({
                "user_id": data.get('user_id'),
                "user_query": data['user_query'],
                "retrieval_items": batch_retrieval_items[i],
                "recommend_item": recommend_items[i],
                "response": responses[i],
            })

        return results
