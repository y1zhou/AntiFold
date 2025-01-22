"""LitServe API server for the AntiFold model."""

from pathlib import Path

import litserve as ls
import numpy as np
import polars as pl
import torch
from pydantic import BaseModel

import antifold.esm
from antifold.antiscripts import load_model, seed_everything
from antifold.esm_util_custom import CoordBatchConverter_mask_gpu
from antifold.if1_dataset import InverseData

MAX_BATCH_SIZE = 16


class InputRequest(BaseModel):
    """Schema for input request parameters."""

    pdb_path: str
    heavy_chain: str
    light_chain: str
    antigen_chain: str


class AntifoldServer(ls.LitAPI):
    """API for Antifold model.

    See https://lightning.ai/docs/litserve/api-reference/litapi/ for details.
    """

    def setup(self, device):
        """Called once at the beginning when the server starts."""
        checkpoint_path = Path(__file__).parent.parent / "models" / "model.pt"
        self.model = load_model(str(checkpoint_path)).eval().to(device)
        self.alphabet = antifold.esm.data.Alphabet.from_architecture("invariant_gvp")
        self.batch_converter = CoordBatchConverter_mask_gpu(self.alphabet)
        self.aa_list = list("ACDEFGHIKLMNPQRSTVWY")

    def decode_request(self, request: InputRequest):
        """Convert incoming request payloads into model-ready inputs."""
        pdb_file_id = Path(request.pdb_path).stem
        return {
            "pdb": pdb_file_id,
            "pdb_path": request.pdb_path,
            "Hchain": request.heavy_chain,
            "Lchain": request.light_chain,
            "antigen_chain": request.antigen_chain,
        }

    def batch(self, inputs: list[dict[str, str]]):
        """Combine inputs from `decode_request` into a single batch."""
        self.log("batch_size", len(inputs))
        return inputs

    def predict(self, inputs: list[dict[str, str]]):
        """Run the model on the (optinally batched) inputs."""
        input_df = pl.from_records(inputs, orient="row")
        dataset, dataloader = self._create_dataset(input_df)

        # In each request there's always only one batch
        batch = next(iter(dataloader))
        all_seq_logits = self._get_logits(batch)

        all_df_logits: list[tuple[str, str, pl.DataFrame]] = []
        for i in range(all_seq_logits.shape[0]):
            seq_probs = all_seq_logits[i]
            # Get PDB sequence, position+insertion code and H+L chain idxs
            pdb_metadata = dataset.pdb_info_dict[i]
            seq = dataset[i][2]
            pdb_res = [aa for aa in seq if aa != "-"]

            # Position + insertion codes - gaps
            posinschain = dataset[i][4]
            posinschain = [p for p in posinschain if p != "nan"]

            # Extract position + insertion code + chain (1-letter)
            pdb_posins = [p[:-1] for p in posinschain]
            pdb_chains = [p[-1] for p in posinschain]

            # Check matches w/ residue probs
            if seq_probs.shape[0] != len(pdb_posins):
                self.log("mismatch_pdb_path", pdb_metadata["pdb_path"])
                # continue

            # Logits to DataFrame
            # Limit to 20x amino-acids probs
            df_logits = pl.from_numpy(
                seq_probs, schema=self.alphabet.all_toks[4:25]
            ).select(
                pl.Series("chain_id", pdb_chains),
                pl.Series("resi", pdb_posins),
                pl.Series("resn", pdb_res),
                *self.aa_list,
            )

            all_df_logits.append(
                (pdb_metadata["pdb_chainsname"], pdb_metadata["pdb_path"], df_logits)
            )

        return all_df_logits

    def unbatch(self, outputs: list[tuple[str, str, pl.DataFrame]]):
        """Split the outputs back into individual responses."""
        return outputs

    def encode_response(self, output: tuple[str, str, pl.DataFrame]):
        """Convert model outputs into a JSON-serializable format."""
        return {
            "pdb_chainsname": output[0],
            "pdb_path": output[1],
            "logits": output[2].to_dict(as_series=False),
        }

    def _create_dataset(self, df: pl.DataFrame):
        """Create a dataset from the input parameters.

        Ref: `antifold.antiscripts.get_dataset_dataloader`
        """
        dat = df.with_columns(
            pl.concat_str("Hchain", "Lchain", "antigen_chain").alias("chain_order")
        ).select(
            "pdb",
            "pdb_path",
            pl.col("chain_order").str.split(""),
            pl.concat_str("pdb", pl.lit("_"), "chain_order").alias("pdb_chainsname"),
            "Hchain",
            "Lchain",
        )
        dataset = InverseData(gaussian_noise_flag=False, custom_chain_mode=True)
        dataset.pdb_info_dict = {i: r for i, r in enumerate(dat.iter_rows(named=True))}

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=MAX_BATCH_SIZE,
            shuffle=False,
            collate_fn=self.batch_converter,
            num_workers=0,
        )
        return dataset, dataloader

    def _get_logits(self, batch):
        """Get logits from the model.

        Ref:

        `antifold.antiscripts.dataset_dataloader_to_predictions_list`
        `antifold.antiscripts.predictions_list_to_df_logits_list`
        """
        seed_everything(42)
        (
            coords,
            confidence,
            strs,
            tokens,
            padding_mask,
            loss_masks,
            res_pos,
            posinschain_list,
            targets,
        ) = batch
        with torch.inference_mode():
            prev_output_tokens = tokens[:, :-1]
            logits, extra = self.model.forward(  # bs x 35 x seq_len, exlude bos, eos
                coords,
                padding_mask,  # Includes masked positions
                confidence,
                prev_output_tokens,
                features_only=False,
            )

            logits = logits.detach().cpu().numpy()
            tokens = tokens.detach().cpu().numpy()

        # List of L x 21 seqprobs (20x AA, 21st == "X")
        mask_gap = tokens[:, 1:] != 30  # 30 is gap
        mask_pad = tokens[:, 1:] != self.alphabet.padding_idx  # 1 is padding
        mask_combined = mask_gap & mask_pad

        # Filter out gap (-) and padding (<pad>) positions, only keep 21x amino-acid probs (4:24) + "X"
        seqprobs_list = [logits[i, 4:25, mask_combined[i]] for i in range(len(logits))]
        return np.stack(seqprobs_list, axis=0)


class SimpleLogger(ls.Logger):
    """Simple logger for the server."""

    def process(self, key, value):
        """Collect messages from the logger queue."""
        print(f"{key}: {value}", flush=True)


if __name__ == "__main__":
    api = AntifoldServer()
    server = ls.LitServer(
        api,
        accelerator="auto",
        max_batch_size=MAX_BATCH_SIZE,
        batch_timeout=0.05,
        timeout=600,
        track_requests=True,
        loggers=SimpleLogger(),
    )
    server.run(host="127.0.0.1", port=8000, generate_client_file=False)
