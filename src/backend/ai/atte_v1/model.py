"""AtteV1 neural network models — attention-based architecture with GRU (design/40).

Provides AtteV1PolicyModel and AtteV1ValueModel.
Architecture: Embedding(e1/e2/e3) → Attention(deck+hand) → GRU → Concat → FC → Heads.
"""

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backend.ai.atte_v1.constants import (
    ACTION_EMBEDDING_DIM, ACTION_VOCAB_SIZE, EMB2_DIM,
    FEATURE_CONCAT_INPUT_SIZE, GRU_INPUT_SIZE, HIDDEN_SIZE,
    ORDINAL_EMBEDDING_DIM, ORDINAL_VOCAB_SIZE, OUTPUT_FEATURE_DIM,
    PAD_ACTION_ID, POLICY_HIDDEN_DIM, POLICY_OUTPUT_SIZE, Q_DIM,
    STATUS_EMBEDDING_DIM, STATUS_VOCAB_SIZE, VALUE_HIDDEN_DIM,
)


# ---------------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------------


class _ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with learnable single query vector (Q1/Q2/Q3)."""

    def __init__(self, q_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, q_dim))
        nn.init.xavier_uniform_(self.query)

    def forward(self, kv: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = kv.shape
        if seq_len == 0:
            return torch.zeros(batch, dim, device=kv.device)
        q = self.query.expand(batch, -1, -1)
        scale = math.sqrt(dim)
        attn = F.softmax(torch.bmm(q, kv.transpose(1, 2)) / scale, dim=-1)
        return torch.bmm(attn, kv).squeeze(1)


class _SelfAttention(nn.Module):
    """Single-head self-attention with residual + LayerNorm (shared for HZ1/HZ2)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 0:
            return x
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scale = math.sqrt(x.shape[-1])
        out = F.scaled_dot_product_attention(q, k, v, scale=scale)
        return self.norm(x + out)


class _CrossAttention(nn.Module):
    """Cross-attention: HZ1 vectors query HZ2 (separate QKV projections)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q1_proj = nn.Linear(dim, dim, bias=False)
        self.k2_proj = nn.Linear(dim, dim, bias=False)
        self.v2_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, hz1: torch.Tensor, hz2: torch.Tensor) -> torch.Tensor:
        if hz1.shape[1] == 0 or hz2.shape[1] == 0:
            return self.norm(hz1)
        q1 = self.q1_proj(hz1)
        k2 = self.k2_proj(hz2)
        v2 = self.v2_proj(hz2)
        scale = math.sqrt(hz1.shape[-1])
        out = F.scaled_dot_product_attention(q1, k2, v2, scale=scale)
        return self.norm(hz1 + out)


# ---------------------------------------------------------------------------
# _AtteV1Body — shared body for Policy and Value models
# ---------------------------------------------------------------------------


class _AtteV1Body(nn.Module):
    """Shared body: embedding + attention + GRU + feature concat → 64-d output."""

    def __init__(self) -> None:
        super().__init__()
        # Embeddings
        self.e1 = nn.Embedding(ACTION_VOCAB_SIZE, ACTION_EMBEDDING_DIM)
        self.e2 = nn.Embedding(STATUS_VOCAB_SIZE, STATUS_EMBEDDING_DIM)
        self.e3 = nn.Embedding(ORDINAL_VOCAB_SIZE, ORDINAL_EMBEDDING_DIM)
        self.emb2_norm = nn.LayerNorm(EMB2_DIM)

        # Attention
        self.q1 = _ScaledDotProductAttention(Q_DIM)
        self.q2 = _ScaledDotProductAttention(Q_DIM)
        self.q3 = _ScaledDotProductAttention(Q_DIM)
        self.self_attn = _SelfAttention(EMB2_DIM)
        self.cross_attn = _CrossAttention(EMB2_DIM)

        # State features
        self.duration_fc = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 16),
        )

        # GRU
        self.gru = nn.GRU(GRU_INPUT_SIZE, HIDDEN_SIZE, num_layers=1, batch_first=False)
        self.h_out_transform = nn.Linear(HIDDEN_SIZE, EMB2_DIM)
        self.h_out_norm = nn.LayerNorm(EMB2_DIM)

        # Feature concat
        self.feature_fc = nn.Sequential(
            nn.Linear(FEATURE_CONCAT_INPUT_SIZE, OUTPUT_FEATURE_DIM), nn.ReLU(),
        )

        self._last_hidden: torch.Tensor | None = None

    def _compute_ordinal_sequences(
        self,
        hand_ids: torch.Tensor,
        discard_ids: torch.Tensor,
        draw_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute cross-sequence ordinals for hand/discard/draw.

        Ordinal = nth occurrence of each action ID across [hand+discard+draw].
        Clipped to [1, ORDINAL_VOCAB_SIZE-1].
        """
        batch, h_len = hand_ids.shape
        d_len = discard_ids.shape[1]
        w_len = draw_ids.shape[1]
        device = hand_ids.device
        max_ord = ORDINAL_VOCAB_SIZE - 1

        # Concatenate all: (batch, h+d+w)
        all_ids = torch.cat([hand_ids, discard_ids, draw_ids], dim=1)
        total = all_ids.shape[1]

        hand_ord = torch.zeros(batch, h_len, dtype=torch.long, device=device)
        disc_ord = torch.zeros(batch, d_len, dtype=torch.long, device=device)
        draw_ord = torch.zeros(batch, w_len, dtype=torch.long, device=device)

        for b in range(batch):
            counts: dict[int, int] = {}
            for t in range(total):
                aid = int(all_ids[b, t].item())
                if aid == 0:
                    continue
                counts[aid] = counts.get(aid, 0) + 1
                cnt = min(counts[aid], max_ord)
                if t < h_len:
                    hand_ord[b, t] = cnt
                elif t < h_len + d_len:
                    disc_ord[b, t - h_len] = cnt
                else:
                    draw_ord[b, t - h_len - d_len] = cnt

        return hand_ord, disc_ord, draw_ord

    def _build_emb2_sequence(
        self,
        hand_ids: torch.Tensor,
        discard_ids: torch.Tensor,
        draw_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Build emb2 for hand+discard+draw with ordinal encoding."""
        batch = hand_ids.shape[0]
        device = hand_ids.device
        hand_ord, disc_ord, draw_ord = self._compute_ordinal_sequences(
            hand_ids, discard_ids, draw_ids,
        )

        parts: list[torch.Tensor] = []
        for ids, ords in [
            (hand_ids, hand_ord), (discard_ids, disc_ord), (draw_ids, draw_ord),
        ]:
            if ids.shape[1] > 0:
                emb1 = self.e1(ids)
                ord_emb = self.e3(ords)
                parts.append(self.emb2_norm(torch.cat([emb1, ord_emb], dim=-1)))
            else:
                parts.append(torch.zeros(batch, 0, EMB2_DIM, device=device))

        return torch.cat(parts, dim=1)

    def _deck_advantage(
        self, own_deck: torch.Tensor, opp_deck: torch.Tensor,
    ) -> torch.Tensor:
        """Z1=Q1(own), Z2=Q1(opp), Z3=Q2([Z1,Z2])."""
        z1 = self.q1(own_deck)
        z2 = self.q1(opp_deck)
        return self.q2(torch.stack([z1, z2], dim=1))

    def _state_features(
        self, status_ids: torch.Tensor, durations: torch.Tensor,
    ) -> torch.Tensor:
        """S' = S * w + b, sum pool → SS(8-d)."""
        batch, seq_len = status_ids.shape
        if seq_len == 0:
            return torch.zeros(batch, 8, device=status_ids.device)
        s_emb = self.e2(status_ids)
        dur_out = self.duration_fc(durations.unsqueeze(-1))  # (b,s,16)
        w, b = dur_out[..., :8], dur_out[..., 8:16]
        return (s_emb * w + b).sum(dim=1)

    def _hand_advantage(
        self,
        h_out_t: torch.Tensor,
        own_hand_emb2: torch.Tensor,
        opp_hand_emb2: torch.Tensor,
    ) -> torch.Tensor:
        """hh1 = [h_out', own_hand]; hh2 = [h_out', opp_hand];
        self-attn → HZ1,HZ2; cross-attn → updated HZ1; Q3 → HAZ."""
        h_exp = h_out_t.unsqueeze(1)
        hh1 = torch.cat([h_exp, own_hand_emb2], dim=1) if own_hand_emb2.shape[1] > 0 else h_exp
        hh2 = torch.cat([h_exp, opp_hand_emb2], dim=1) if opp_hand_emb2.shape[1] > 0 else h_exp
        hz1 = self.self_attn(hh1)
        hz2 = self.self_attn(hh2)
        hz1 = self.cross_attn(hz1, hz2)
        return self.q3(hz1)

    def forward(
        self,
        own_hand_ids: torch.Tensor,
        own_discard_ids: torch.Tensor,
        own_draw_ids: torch.Tensor,
        own_last_action: torch.Tensor,
        own_status_ids: torch.Tensor,
        own_status_durations: torch.Tensor,
        own_opp_hand_ids: torch.Tensor,
        own_opp_discard_ids: torch.Tensor,
        own_opp_draw_ids: torch.Tensor,
        opp_hand_ids: torch.Tensor,
        opp_discard_ids: torch.Tensor,
        opp_draw_ids: torch.Tensor,
        opp_last_action: torch.Tensor,
        opp_status_ids: torch.Tensor,
        opp_status_durations: torch.Tensor,
        own_hp: torch.Tensor,
        own_energy: torch.Tensor,
        opp_hp: torch.Tensor,
        opp_energy: torch.Tensor,
        round_in_cycle: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        batch = own_hand_ids.shape[0]

        # Map last_action -1 → PAD
        own_la = own_last_action.clone()
        own_la[own_la == -1] = PAD_ACTION_ID
        opp_la = opp_last_action.clone()
        opp_la[opp_la == -1] = PAD_ACTION_ID

        own_la_emb = self.e1(own_la).squeeze(1)
        opp_la_emb = self.e1(opp_la).squeeze(1)

        # Deck advantage
        own_deck = self._build_emb2_sequence(own_hand_ids, own_discard_ids, own_draw_ids)
        opp_deck = self._build_emb2_sequence(own_opp_hand_ids, own_opp_discard_ids, own_opp_draw_ids)
        z3 = self._deck_advantage(own_deck, opp_deck)

        # State features
        own_ss = self._state_features(own_status_ids, own_status_durations)
        opp_ss = self._state_features(opp_status_ids, opp_status_durations)

        # GRU
        gru_in = torch.cat([own_la_emb, opp_la_emb, own_ss, opp_ss], dim=-1).unsqueeze(0)
        gru_out, new_h = self.gru(gru_in, hidden)
        h_out = gru_out.squeeze(0)
        self._last_hidden = new_h
        h_out_t = self.h_out_norm(self.h_out_transform(h_out))

        # Hand advantage
        own_hand_emb2 = self._build_emb2_sequence(
            own_hand_ids,
            torch.zeros(batch, 0, dtype=torch.long, device=own_hand_ids.device),
            torch.zeros(batch, 0, dtype=torch.long, device=own_hand_ids.device),
        )
        opp_hand_emb2 = self._build_emb2_sequence(
            own_opp_hand_ids,
            torch.zeros(batch, 0, dtype=torch.long, device=own_hand_ids.device),
            torch.zeros(batch, 0, dtype=torch.long, device=own_hand_ids.device),
        )
        haz = self._hand_advantage(h_out_t, own_hand_emb2, opp_hand_emb2)

        # Feature concat — ensure all are (batch, d) before cat
        feats = torch.cat([
            own_hp.reshape(batch, 1), own_energy.reshape(batch, 1),
            opp_hp.reshape(batch, 1), opp_energy.reshape(batch, 1),
            round_in_cycle.reshape(batch, 1),
            own_la_emb, opp_la_emb, haz, z3, h_out,
        ], dim=-1)
        return self.feature_fc(feats)


# ---------------------------------------------------------------------------
# AtteV1PolicyModel
# ---------------------------------------------------------------------------


class AtteV1PolicyModel(nn.Module):
    """AtteV1 Policy network. Policy Head FC(32→24) shares weight with e1[:24]."""

    INITIAL_HIDDEN_STATE: list[float] = [0.0] * HIDDEN_SIZE

    def __init__(self) -> None:
        super().__init__()
        self.body = _AtteV1Body()
        self.fc1 = nn.Linear(OUTPUT_FEATURE_DIM, POLICY_HIDDEN_DIM)
        self.fc_policy = nn.Linear(POLICY_HIDDEN_DIM, POLICY_OUTPUT_SIZE)
        # Tie fc_policy.weight to e1.weight[:24]
        self.fc_policy.weight = nn.Parameter(self.body.e1.weight[:24].detach().clone())

    @property
    def e1(self) -> nn.Embedding:
        """Expose e1 for test inspection."""
        return self.body.e1

    def forward(
        self,
        own_hand_ids: torch.Tensor,
        own_discard_ids: torch.Tensor,
        own_draw_ids: torch.Tensor,
        own_last_action: torch.Tensor,
        own_status_ids: torch.Tensor,
        own_status_durations: torch.Tensor,
        own_opp_hand_ids: torch.Tensor,
        own_opp_discard_ids: torch.Tensor,
        own_opp_draw_ids: torch.Tensor,
        opp_hand_ids: torch.Tensor,
        opp_discard_ids: torch.Tensor,
        opp_draw_ids: torch.Tensor,
        opp_last_action: torch.Tensor,
        opp_status_ids: torch.Tensor,
        opp_status_durations: torch.Tensor,
        own_hp: torch.Tensor,
        own_energy: torch.Tensor,
        opp_hp: torch.Tensor,
        opp_energy: torch.Tensor,
        round_in_cycle: torch.Tensor,
        hidden: torch.Tensor,
        mask_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (probs, log_probs, new_hidden)."""
        out = self.body(
            own_hand_ids, own_discard_ids, own_draw_ids,
            own_last_action, own_status_ids, own_status_durations,
            own_opp_hand_ids, own_opp_discard_ids, own_opp_draw_ids,
            opp_hand_ids, opp_discard_ids, opp_draw_ids,
            opp_last_action, opp_status_ids, opp_status_durations,
            own_hp, own_energy, opp_hp, opp_energy,
            round_in_cycle, hidden,
        )
        h1 = torch.relu(self.fc1(out))
        logits = self.fc_policy(h1)
        if mask_fn is not None:
            logits = mask_fn(logits)
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        log_probs = F.log_softmax(logits, dim=-1)
        return torch.exp(log_probs), log_probs, self.body._last_hidden


# ---------------------------------------------------------------------------
# AtteV1ValueModel
# ---------------------------------------------------------------------------


class AtteV1ValueModel(nn.Module):
    """AtteV1 Value network. Value Head FC(32→1), no tanh."""

    INITIAL_HIDDEN_STATE: list[float] = [0.0] * HIDDEN_SIZE

    def __init__(self) -> None:
        super().__init__()
        self.body = _AtteV1Body()
        self.fc1 = nn.Linear(OUTPUT_FEATURE_DIM, VALUE_HIDDEN_DIM)
        self.fc_value = nn.Linear(VALUE_HIDDEN_DIM, 1)

    def forward(
        self,
        own_hand_ids: torch.Tensor,
        own_discard_ids: torch.Tensor,
        own_draw_ids: torch.Tensor,
        own_last_action: torch.Tensor,
        own_status_ids: torch.Tensor,
        own_status_durations: torch.Tensor,
        own_opp_hand_ids: torch.Tensor,
        own_opp_discard_ids: torch.Tensor,
        own_opp_draw_ids: torch.Tensor,
        opp_hand_ids: torch.Tensor,
        opp_discard_ids: torch.Tensor,
        opp_draw_ids: torch.Tensor,
        opp_last_action: torch.Tensor,
        opp_status_ids: torch.Tensor,
        opp_status_durations: torch.Tensor,
        own_hp: torch.Tensor,
        own_energy: torch.Tensor,
        opp_hp: torch.Tensor,
        opp_energy: torch.Tensor,
        round_in_cycle: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (value, new_hidden). No tanh."""
        out = self.body(
            own_hand_ids, own_discard_ids, own_draw_ids,
            own_last_action, own_status_ids, own_status_durations,
            own_opp_hand_ids, own_opp_discard_ids, own_opp_draw_ids,
            opp_hand_ids, opp_discard_ids, opp_draw_ids,
            opp_last_action, opp_status_ids, opp_status_durations,
            own_hp, own_energy, opp_hp, opp_energy,
            round_in_cycle, hidden,
        )
        h1 = torch.relu(self.fc1(out))
        return self.fc_value(h1), self.body._last_hidden
