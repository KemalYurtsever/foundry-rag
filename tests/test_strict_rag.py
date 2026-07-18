from __future__ import annotations

import unittest

from foundry_rag.documents import Chunk, chunk_text
from foundry_rag.generators import NO_ANSWER, Evidence, EvidenceSelection
from foundry_rag.pipeline import RAGPipeline
from foundry_rag.retriever import HybridRetriever, content_tokens


class FixedGenerator:
    context_length = None
    max_output_tokens = None

    def __init__(self, selection: EvidenceSelection):
        self.selection = selection

    def generate(self, question, results):
        return self.selection

    def close(self):
        return None


class StrictRAGTests(unittest.TestCase):
    def test_unsupported_question_returns_no_answer(self):
        chunk = Chunk("chunk-car", "car.txt", "The car has four wheels.")
        pipeline = RAGPipeline(HybridRetriever([chunk]))

        answer = pipeline.ask("What color is the car?", top_k=1)

        self.assertEqual(answer.text, NO_ANSWER)
        self.assertEqual(answer.sources, ())

    def test_supported_question_returns_only_exact_quote(self):
        chunk = Chunk("chunk-car", "car.txt", "The car color is red.")
        pipeline = RAGPipeline(HybridRetriever([chunk]))

        answer = pipeline.ask("What color is the car?", top_k=1)

        self.assertEqual(answer.text, "The car color is red.")
        self.assertEqual(answer.sources, ("car.txt",))

    def test_uncited_model_output_is_rejected(self):
        chunk = Chunk("chunk-car", "car.txt", "The car color is red.")
        generator = FixedGenerator(EvidenceSelection(answerable=True, evidence=()))
        pipeline = RAGPipeline(HybridRetriever([chunk]), generator)

        answer = pipeline.ask("What color is the car?", top_k=1)

        self.assertEqual(answer.text, NO_ANSWER)

    def test_non_verbatim_model_quote_is_rejected(self):
        chunk = Chunk("chunk-car", "car.txt", "The car color is red.")
        generator = FixedGenerator(
            EvidenceSelection(
                answerable=True,
                evidence=(Evidence("chunk-car", "The car is blue."),),
            )
        )
        pipeline = RAGPipeline(HybridRetriever([chunk]), generator)

        answer = pipeline.ask("What color is the car?", top_k=1)

        self.assertEqual(answer.text, NO_ANSWER)

    def test_irrelevant_exact_quote_is_rejected(self):
        chunk = Chunk("chunk-car", "car.txt", "The car has four wheels.")
        generator = FixedGenerator(
            EvidenceSelection(
                answerable=True,
                evidence=(Evidence("chunk-car", "The car has four wheels."),),
            )
        )
        pipeline = RAGPipeline(HybridRetriever([chunk]), generator)

        answer = pipeline.ask("What color is the car?", top_k=1)

        self.assertEqual(answer.text, NO_ANSWER)


    def test_answers_follow_document_even_when_factually_wrong(self):
        chunk = Chunk("chunk-flag", "flag.txt", "The sample flag is orange.")
        pipeline = RAGPipeline(HybridRetriever([chunk]))

        answer = pipeline.ask("What color is the sample flag?", top_k=1)

        self.assertEqual(answer.text, "The sample flag is orange.")
        self.assertEqual(answer.sources, ("flag.txt",))

    def test_color_question_does_not_accept_unrelated_subject_quote(self):
        chunk = Chunk("chunk-car", "car.txt", "The car has four wheels.")
        pipeline = RAGPipeline(HybridRetriever([chunk]))

        answer = pipeline.ask("What color is the car?", top_k=1)

        self.assertEqual(answer.text, NO_ANSWER)
        self.assertEqual(answer.sources, ())


    def test_abstention_includes_diagnostics(self):
        chunk = Chunk("chunk-car", "car.txt", "The car has four wheels.")
        pipeline = RAGPipeline(HybridRetriever([chunk]))

        answer = pipeline.ask("What color is the car?", top_k=1)

        self.assertEqual(answer.text, NO_ANSWER)
        self.assertIn("no validated evidence", answer.diagnostics)

    def test_ai_is_not_removed_as_a_stopword(self):
        self.assertEqual(content_tokens("What is AI?"), ["ai"])
        chunk = Chunk(
            "chunk-ai",
            "ai.txt",
            "AI is artificial intelligence.",
        )
        pipeline = RAGPipeline(HybridRetriever([chunk]))

        answer = pipeline.ask("What is AI?", top_k=1)

        self.assertEqual(answer.text, "AI is artificial intelligence.")

    def test_zero_scores_do_not_receive_rrf_rank(self):
        self.assertEqual(HybridRetriever._positive_rank([0.0, 0.0]), [])
        self.assertEqual(HybridRetriever._positive_rank([0.0, 0.2, 0.1]), [1, 2])

    def test_chunking_preserves_sentence_punctuation(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, "sample.txt", chunk_size=24, overlap=4)
        combined = " ".join(chunk.text for chunk in chunks)

        self.assertIn("First sentence.", combined)
        self.assertIn("Second sentence.", combined)
        self.assertIn("Third sentence.", combined)

    def test_comparison_returns_multiple_grounded_sections(self):
        chunks = [
            Chunk(
                "strengths",
                "similar_systems.txt",
                "Strengths\n- Integration: Existing partnerships simplify deployment.\n"
                "- Archives: Long-term records support analysis.",
            ),
            Chunk(
                "weaknesses",
                "similar_systems.txt",
                "Weaknesses\n- Obstruction: Physical obstacles can block operation.\n"
                "- Interference: Congestion can disrupt connectivity.",
            ),
        ]
        pipeline = RAGPipeline(HybridRetriever(chunks), neighbor_window=1)

        answer = pipeline.ask(
            "What are the strengths and weaknesses of similar systems?", top_k=2
        )

        self.assertIn("Integration", answer.text)
        self.assertIn("Archives", answer.text)
        self.assertIn("Obstruction", answer.text)
        self.assertIn("Interference", answer.text)
    def test_invalid_top_k_raises(self):
        chunk = Chunk("chunk-one", "one.txt", "One fact.")
        pipeline = RAGPipeline(HybridRetriever([chunk]))
        with self.assertRaises(ValueError):
            pipeline.ask("One?", top_k=0)


if __name__ == "__main__":
    unittest.main()


