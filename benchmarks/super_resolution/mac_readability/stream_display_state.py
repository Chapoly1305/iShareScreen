"""Ownership of displayed content in the explicit-source-generation protocol.

Decoded codec revisions may change while the source pixels are unchanged.
Only an authenticated sender's new source announcement changes that ownership.
This assumption is specific to this experiment's protocol, not ordinary video.
"""


class DisplayState:
    def __init__(self, policy="pixel"):
        if policy not in ("pixel", "hold", "source"):
            raise ValueError("Unknown display policy")
        self.policy = policy
        self.generation = 0
        self.phase = "blank"
        self.sr_revision = -1

    def begin(self, generation):
        if generation <= self.generation:
            raise ValueError("Source generation must increase")
        self.generation = generation
        self.phase = "blank"
        self.sr_revision = -1

    def base(self, generation):
        if generation != self.generation or self.phase == "detail":
            return False
        if self.policy != "pixel" and self.phase == "sr":
            return False
        self.phase = "base"
        return True

    def sr_rejection(
        self, generation, input_revision, current_revision, source_backed=False
    ):
        if generation != self.generation:
            return "source_changed"
        if source_backed and self.phase != "detail":
            return "source_detail_unavailable"
        if self.phase == "detail" and not source_backed:
            return "exact_source_already_visible"
        if input_revision <= self.sr_revision:
            return "older_sr_revision"
        if self.policy != "source" and input_revision != current_revision:
            return "codec_revision_changed"
        return None

    def sr(self, generation, input_revision, current_revision, source_backed=False):
        reason = self.sr_rejection(
            generation, input_revision, current_revision, source_backed
        )
        if reason:
            return False
        self.phase = "detail" if source_backed else "sr"
        self.sr_revision = input_revision
        return True

    def detail(self, generation):
        if generation != self.generation:
            return False
        self.phase = "detail"
        return True
