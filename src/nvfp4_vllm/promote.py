"""Sealing a filled BF16 tail page into the FP4 cache.

A sequence's trailing tokens live in BF16 because V is packed along the token
axis a whole page at a time, so a partial page cannot be quantized. The step
that brings a tail to exactly one page removes that obstacle, and the page
moves into the block vLLM already reserved for it. Nothing else in the design
ever writes an FP4 page after a decode step.

One launch per step, always, whether or not any row crossed a boundary. The
alternative — asking the device which rows crossed and launching only then —
costs a synchronization every step to save work on 127 steps out of 128, and
those 128 steps have to pay for the answer anyway. What makes the fixed launch
affordable is that it covers all the layers, K and V at once: one per layer
would cost dozens of launches a step against roughly one millisecond of
attention.

Timing is the whole correctness argument here. This runs after the last
layer's attention, by which point every layer has read the tail as this step's
metadata describes it and written its own share of the token that filled the
page. Running it a layer early would quantize a page that a later layer has
not finished writing, and the loss would be invisible until it showed up as
slightly worse output.
"""

from __future__ import annotations


def launch(metadata, runtime) -> None:
    """Quantize every filled tail page into its block, for every layer.

    ``promotion_pages`` is the whole slot table rather than this step's rows,
    so the launch shape is a constant of the engine. Rows that did not fill a
    page carry -1 and are skipped inside the kernel.

    The runtime's promotion writer is the write path's kernel with only its
    page roles: K and V pages, no tail, and no rotation or shift, since the
    tail already holds K in the cache's basis. Every layer's own destination
    comes from the base-address table it was built with, because a KV cache is
    allocated a layer at a time and the layers are not a stride apart.
    """
    writer, regions = runtime.promotion_writer()
    writer(
        runtime.tail_key_tokens,
        runtime.tail_value_tokens,
        regions,
        metadata.promotion_source_tokens,
        metadata.promotion_pages,
    )
