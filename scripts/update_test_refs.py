#!/usr/bin/env python3
"""Auto-update Scala test reference arrays from test_vectors.json.

Reads export/test_vectors/test_vectors.json and updates:
  - ZyboGPTRomeoSim.scala: referenceGenerated array
  - ZyboGPTPipelineDebugSim.scala: all 16 refAfter* arrays, expectedToken, input token
"""

import json
import os
import re
import sys


def format_scala_array(values, name, indent=2):
    """Format a list of ints as a Scala Array(...) assignment."""
    prefix = " " * indent
    vals_per_line = 16
    lines = []
    for i in range(0, len(values), vals_per_line):
        chunk = values[i:i + vals_per_line]
        lines.append(", ".join(str(v) for v in chunk))
    if len(lines) == 1:
        return f"{prefix}val {name} = Array({lines[0]})"
    else:
        inner = (",\n" + prefix + "    ").join(lines)
        return f"{prefix}val {name} = Array(\n{prefix}    {inner})"


def replace_scala_array(content, var_name, new_values):
    """Replace a val <var_name> = Array(...) declaration in Scala source."""
    # Match: val <var_name> = Array(<anything, possibly multiline>)
    pattern = rf'([ \t]*)val {re.escape(var_name)} = Array\([^)]*\)'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"  WARNING: Could not find 'val {var_name} = Array(...)' in source")
        return content

    indent = len(match.group(1))
    new_decl = format_scala_array(new_values, var_name, indent)
    return content[:match.start()] + new_decl + content[match.end():]


def update_romeo_sim(test_vectors, scala_path):
    """Update ZyboGPTRomeoSim.scala referenceGenerated array."""
    prompt = test_vectors["prompt"]
    # referenceGenerated starts from the prediction after the full prompt:
    # generated[len(prompt)-1] = prediction after processing the last prompt token
    generated = test_vectors["generated"][len(prompt) - 1:]

    with open(scala_path, "r") as f:
        content = f.read()

    # Update referenceGenerated array
    content = replace_scala_array(content, "referenceGenerated", generated)

    # Update the training comment line (line before referenceGenerated)
    content = re.sub(
        r'// Model retrained with .*',
        '// Model retrained with two-phase curriculum (float pretrain + hw-mode fine-tune)',
        content,
    )

    with open(scala_path, "w") as f:
        f.write(content)
    print(f"  Updated {scala_path}")


def update_pipeline_debug_sim(test_vectors, scala_path):
    """Update ZyboGPTPipelineDebugSim.scala reference arrays."""
    debug = test_vectors["debug"]

    # Step 6 = first autoregressive step (position 6)
    step6 = debug[6]

    # Stage name mapping: test_vectors key -> Scala variable name
    stage_map = {
        "after_embed": "refAfterEmbed",
        "after_layer0_attn_norm": "refAfterLayer0AttnNorm",
        "after_layer0_q_proj": "refAfterLayer0QProj",
        "after_layer0_k_proj": "refAfterLayer0KProj",
        "after_layer0_v_proj": "refAfterLayer0VProj",
        "after_layer0_attn_out": "refAfterLayer0AttnOut",
        "after_layer0_ff_norm": "refAfterLayer0FfNorm",
        "after_layer0_ff_down": "refAfterLayer0FfDown",
        "after_layer1_attn_norm": "refAfterLayer1AttnNorm",
        "after_layer1_q_proj": "refAfterLayer1QProj",
        "after_layer1_k_proj": "refAfterLayer1KProj",
        "after_layer1_v_proj": "refAfterLayer1VProj",
        "after_layer1_attn_out": "refAfterLayer1AttnOut",
        "after_layer1_ff_norm": "refAfterLayer1FfNorm",
        "after_layer1_ff_down": "refAfterLayer1FfDown",
        "after_final_norm": "refAfterFinalNorm",
    }

    with open(scala_path, "r") as f:
        content = f.read()

    # Update all 16 reference arrays
    for json_key, scala_var in stage_map.items():
        if json_key in step6:
            content = replace_scala_array(content, scala_var, step6[json_key])
        else:
            print(f"  WARNING: '{json_key}' not found in step 6 debug data")

    # Update expectedToken (prediction at position 6)
    expected_token = step6["next_token"]
    content = re.sub(
        r'val expectedToken = \d+.*',
        f'val expectedToken = {expected_token}',
        content,
    )

    # Update the model comment
    content = re.sub(
        r'// Model: .*',
        '// Model: two-phase curriculum (float pretrain + hw-mode fine-tune)',
        content,
    )

    # Update input token on line with axiWrite(0x08, ...)
    # This is the prediction at position 5 (debug[5]["next_token"])
    input_token = debug[5]["next_token"]
    content = re.sub(
        r'axiWrite\(0x08, \d+L\)(\s*)//.*',
        f'axiWrite(0x08, {input_token}L)\\1// token = {input_token}',
        content,
    )

    # Update the print message that says the token value
    if 32 <= input_token < 127:
        token_repr = f"'{chr(input_token)}'"
    elif input_token == 10:
        token_repr = r"'\n'"  # Escaped for Scala string literal
    else:
        token_repr = f"0x{input_token:02x}"
    # Match both correct (\n) and broken (literal newline) versions of the println
    new_println = 'println("\\n--- Starting position 6 (token={} {}) with pipeline monitoring ---")'.format(
        input_token, token_repr)
    content = re.sub(
        r'println\("(?:\\n|\n)--- Starting position 6 \(token=\d+ .*?\) with pipeline monitoring ---"\)',
        lambda m: new_println,
        content,
    )

    # Update the comment at the top about which token/position
    content = re.sub(
        r'// \(token=\d+ .*?, position=6',
        f'// (token={input_token}, position=6',
        content,
    )

    with open(scala_path, "w") as f:
        f.write(content)
    print(f"  Updated {scala_path}")


def main():
    # Find project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Paths
    test_vectors_path = os.path.join(project_root, "export", "test_vectors", "test_vectors.json")
    romeo_sim_path = os.path.join(
        project_root, "hw", "src", "test", "scala", "zybogpt", "ZyboGPTRomeoSim.scala"
    )
    pipeline_debug_path = os.path.join(
        project_root, "hw", "src", "test", "scala", "zybogpt", "ZyboGPTPipelineDebugSim.scala"
    )

    # Allow override via CLI
    if len(sys.argv) > 1:
        test_vectors_path = sys.argv[1]

    if not os.path.exists(test_vectors_path):
        print(f"ERROR: Test vectors not found at {test_vectors_path}")
        print("Run 'make validate' first to generate test vectors.")
        sys.exit(1)

    print(f"Loading test vectors from {test_vectors_path}")
    with open(test_vectors_path) as f:
        test_vectors = json.load(f)

    print(f"  Prompt: {test_vectors['prompt']}")
    print(f"  Generated: {test_vectors['generated'][:20]}...")
    print(f"  Debug steps: {len(test_vectors['debug'])}")

    print("\nUpdating Scala test files:")
    update_romeo_sim(test_vectors, romeo_sim_path)
    update_pipeline_debug_sim(test_vectors, pipeline_debug_path)

    print("\nDone! Run 'make spinal-test' and 'make pipeline-debug' to verify.")


if __name__ == "__main__":
    main()
