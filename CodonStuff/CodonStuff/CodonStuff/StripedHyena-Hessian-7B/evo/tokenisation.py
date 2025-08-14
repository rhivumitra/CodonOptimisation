# The purpose of this Python script is to define the functions required
# for tokenising amino acid as well as nucleotide sequences

from math import ceil
import torch
import json
from torch.nn.utils.rnn import pad_sequence
import Levenshtein


# import tensorflow as tf
import numpy as np
# import keras_nlp
class Tokenizer:
    def __init__(self):
        # # Amino acid tokenization parameters
        # self.ALL_AAS = 'ACDEFGHIKLMNPQRSTVWY*'
        # self.ADDITIONAL_TOKENS = ['<PAD>', '<START>', '<END>','<SEP>', '<UNK>']
        # self.ADDED_TOKENS_PER_SEQ = 2
        # self.eod_id = 2  # End-of-document token ID
        # self.eos_id = 2  # End-of-sequence token ID

        # # Create amino acid token mappings
        # self.n_aas = len(self.ALL_AAS)
        # self.n_add_aa_tokens = len(self.ADDITIONAL_TOKENS)
        # self.aa_token_to_index = {
        #     aa: i + self.n_add_aa_tokens for i, aa in enumerate(self.ALL_AAS)
        # }
        # self.additional_aa_token_to_index = {
        #     token: i for i, token in enumerate(self.ADDITIONAL_TOKENS)
        # }
        # self.global_aa_token_to_index = {
        #     **self.additional_aa_token_to_index, **self.aa_token_to_index
        # }
        # self.index_to_aa_token = {
        #     index: token for token, index in self.global_aa_token_to_index.items()
        # }
        # self.N_AA_TOKENS = len(self.global_aa_token_to_index)

        # # Nucleotide/codon tokenization parameters
        # self.ALL_CODONS = [
        #     nt1 + nt2 + nt3 for nt1 in "ATCG" for nt2 in "ATCG" for nt3 in "ATCG"
        # ]
        # self.ADDITIONAL_NT_TOKENS = ['<PAD>', '<START>', '<END>','<SEP>', '<UNK>']


        # # Create codon token mappings
        # self.n_codons = len(self.ALL_CODONS)
        # self.n_additional_tokens = len(self.ADDITIONAL_NT_TOKENS)
        # self.CODON_OFFSET = 26
        # self.codon_token_to_index = {
        #     codon: i + self.CODON_OFFSET for i, codon in enumerate(self.ALL_CODONS)
        #     #codon: i + self.n_additional_tokens for i, codon in enumerate(self.ALL_CODONS)
        # }
        # self.additional_codon_token_to_index = {
        #     token: i for i, token in enumerate(self.ADDITIONAL_NT_TOKENS)
        # }
        # self.global_codon_token_to_index = {
        #     **self.additional_codon_token_to_index, **self.codon_token_to_index
        # }
        # self.index_to_codon_token = {
        #     index: token for token, index in self.global_codon_token_to_index.items()
        # }
        # self.N_CODON_TOKENS = len(self.global_codon_token_to_index)

        # Load unified vocab
        vocab_path = "/home/rm15weti/CodonStuff/CodonStuff/CodonStuff/StripedHyena-Hessian-7B/evo/tokenizer/full_biotokenizer.json"
        with open(vocab_path, "r") as f:
            self.token_to_index = json.load(f)

        self.index_to_token = {v: k for k, v in self.token_to_index.items()}
        self.vocab_size = len(self.token_to_index)

        # Special token IDs (if needed explicitly)
        self.pad_token = "<PAD>"
        self.sep_token = "<SEP>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"

        self.pad_token_id = self.token_to_index.get(self.pad_token)
        self.sep_token_id = self.token_to_index.get(self.sep_token)
        self.start_token_id = self.token_to_index.get(self.start_token)
        self.end_token_id = self.token_to_index.get(self.end_token)
        self.unk_token_id = self.token_to_index.get(self.unk_token)

        # Optional, if you're distinguishing AA and codon tokens:
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY*'
        self.codons = [nt1 + nt2 + nt3 for nt1 in "ATCG" for nt2 in "ATCG" for nt3 in "ATCG"]

        # Derived vocab segments (use only if needed)
        self.aa_tokens = [aa for aa in self.amino_acids if aa in self.token_to_index]
        self.codon_tokens = [codon for codon in self.codons if codon in self.token_to_index]
        self.special_tokens = [tok for tok in [self.pad_token, self.start_token, self.end_token, self.sep_token, self.unk_token] if tok in self.token_to_index]      




        # Tokenisation must be performed for the "language" of amino acids as 
        # well as for the "language" of nucleotides
        # For both languages, tokenisation is performed via mapping of the
        # individual tokens to integer indices

        # Start with the preparation of amino acid tokenisation

        # The enumeration of amino acids below does not contain selenocysteine
        # (U) and the letter X, denoting any amino acid / an unknown amino acid
        # This is due to the fact that the coding sequences were downloaded from
        # NCBI, not the protein sequences, and nucleotide sequences containing
        # ambiguous characters (such as N for any nucleotide) were excluded
        # Hence, the amino acid character X does not occur
        # Moreover, the default table from NCBI was used for translation, i. e.
        # the amino acid selenocysteine does not occur in the protein sequences
        # However, the asterisk is included, as it denotes a stop codon
        # ALL_AAS = 'ACDEFGHIKLMNPQRSTVWY*'
        # Compared to the tokenisation implemented in ProteinBERT, the '<OTHER>'
        # token is omitted as an entirely unambiguous alphabet is employed for
        # both the nucleotide sequences and the amino acid sequennces
        # The padding token is usually mapped to the integer zero, which is why
        # this is done here as well
        # The consequences of mapping the padding token to another integer or
        # whether this has any consequences at all is not known
        #ADDITIONAL_TOKENS = ['<PAD>', '<START>', '<END>']

        # To each sequence, one <START> and <END> token is added, respectively
        #ADDED_TOKENS_PER_SEQ = 2

    # @property
    # def eod(self):
    #     return self.eod_id

    # # duplicate to suppose both names, eos and eod
    # @property
    # def eos(self):
    #     return self.eod_id

    # @property
    # def vocab_size(self):
    #     """
    #     This property returns the size of the vocabulary, i. e. the number of
    #     unique tokens in the amino acid "language" plus the number of
    #     additional tokens, such as <START>, <END>, etc.
    #     """
    #     return len(self.N_AA_TOKENS) + len(self.ADDITIONAL_TOKENS)
    
    def parse_seq(self, seq):
        """
        This function is responsible for converting an input sequence to a
        string, if applicable, and to return the input string otherwise.

        Parameters
        ----------
        seq: str or binary
            The input amino acid sequence to be tokenised.

        Returns
        -------
        seq: str
            The input amino acid sequence to be tokenised.
        """
        if isinstance(seq, str):
            return seq
        elif isinstance(seq, bytes):
            return seq.decode('utf8')
        else:
            raise TypeError('Unexpected sequence type: %s' % type(seq))


    def tokenise_aa_seq(self, seq, add_special_tokens=True):
        """
        This function performs tokenisation for amino acid sequences. This
        is achieved by splitting an amino acid sequence into individual
        tokens, i. e. individual amino acids, and replacing all individual 
        tokens with their unique integer index defined in the dictionary
        `token_to_index`.

        Apart from that, one <START> and <END> token is added to each
        sequence, respectively. The token dictionary also comprises a
        special token, <OTHER>, reserved for unknown amino acids.

        This function employs the private function `parse_seq`.

        Parameters
        ----------
        seq:  str or binary
            The amino acid sequence to be tokenised.

        Returns
        -------
        tokenised_seq: list, dtype=int
            The tokenised analogue of the input amino acid sequence.
        """
        # It is important to prepend the <START> token and append <END>
        # token to the sequence
        # tokens = [
        #     self.aa_token_to_index.get(aa)
        #     for aa in self.parse_seq(seq)
        # ]
        tokens = [
            self.token_to_index.get(aa)
            for aa in self.parse_seq(seq)
        ]
        if add_special_tokens:
            tokens = (
                [self.start_token_id]
                + tokens
                #+ [self.end_token_id]
            )
        return tokens


    def tokenise_aa_seqs(self, seqs):
        """
        This function performs tokenisation for a whole data set of amino
        acid sequences.

        For this purpose, it employs the private function
        `tokenise_aa_seq`.

        Parameters
        ----------
        seqs: iterable, dtype=str or dtype=binary
            An iterable containing the amino acid sequences to be tokenised.

        Returns
        -------
        tokenised_seqs_array: ndarray, dtype=int32, shape=(m, n)
            A NumPy array harbouring the tokenised analogues of the input
            amino acid sequences. The first dimension, m, corresponds to the
            amount of input sequences, whereas the second dimension, n,
            corresponds to the largest sequence length in the batch.
        """
        # First, perform the mere tokenisation
        tokenised_seqs_list = []

        for seq in seqs:
            for token in seq:
                index = self.token_to_index.get(token)
                if index is None:
                    raise ValueError(f"Unknown amino acid token: {token} in sequence {seq}")
                    break
            tokenised_seqs_list.append(self.tokenise_aa_seq(seq))

        # Now, compute the largest length of the tokenised sequences
        # comprised in the batch
        # Computing the largest length of the tokenised (integer) sequences
        # instead of the plain string sequences is necessary as in the
        # course of tokenisation, the length of each sequence is incremented
        # by two due to the addition of the <START> and <END> token
        max_length = max(map(len, tokenised_seqs_list))

        # Perform padding for each of the tokenised sequences
        tokenised_seqs_array = np.array(
            [tokenised_seq
            +
            (max_length - len(tokenised_seq))*[self.pad_token_id]
            for tokenised_seq in tokenised_seqs_list],
            dtype=np.int32
            )

        #return tokenised_seqs_array
        return tokenised_seqs_list


    # Now, prepare nucleotide/codon tokenisation
    # As it is exclusively dealt with DNA sequences, and not RNA sequences,
    # uracil is omitted
    # The letter N, denoting any nucleotide, is not included as sequences
    # containing ambiguous nucleotides were discarded in advance in order to
    # permit an unambiguous translation into amino acids
    # Note that in the case of nucleotide sequences, tokenisation is not
    # performed for the individual characters/nucleotides as with the amino
    # acids, but rather for individual codons, i. e. triplets

    # As with the amino acid tokenisation, the '<OTHER>' token is omitted as
    # an entirely unambiguous alphabet is used for the nucleotide
    # tokenisation and sequences containing ambiguous characters have been
    # discarded
    # Again, the padding token is deliberately mapped to the integer zero as
    # the consequences of deviating from this convention are not known

    @property
    def vocab_size_NT(self):
        # Example for codons
        return len(self.ALL_CODONS) + len(self.ADDITIONAL_TOKENS)
    
    def _split_into_triplets(self, nt_seq):
        """
        This function splits an input nucleotide sequence into its
        individual codons / triplets.

        The function assumes that the sequence length is a multiple of three
        and that the first and last codon is the AUG start codon and one of
        the three stop codons, respectively.

        Parameters
        ----------
        nt_seq: str
            The nucleotide sequence to be split into triplets / codons.

        Returns
        -------
        codons: list, dtype=str
            The list containing the codons the input nucleotide sequence is
            made up of.
        """
        codons = []
        n_triplets = len(nt_seq) // 3

        for i in range(n_triplets):
            start_index = 3 * i
            end_index = 3 * (i + 1)
            triplet = nt_seq[start_index:end_index]
            codons.append(triplet)

        return codons


    def _tokenise_nt_seq(self, seq, add_special_tokens=True):
        """
        This function is analogous to the function `tokenise_aa_seq` and
        performs tokenisation for nucleotide sequences. However, instead of
        taking the individual nucleotide letters as tokens and mapping them
        to integer indices, three successive nucleotides, i. e. triplets /
        codons are taken as tokens. Hence, there are 64 unique tokens,
        without the additional tokens (e. g. <START>, <END>, etc.).

        For this purpose, it employs the private function
        `_split_into_triplets`.

        Parameters
        ----------
        seq: str or binary
            The input nucleotide sequence to be tokenised.

        Returns
        -------
        tokenised_seq: list, dtype=int
            The tokenised analogue of the input nucleotide sequence.
        """
        # It is important to prepend the <START> token and append the <END>
        # token to the sequence
        # tokens = [
        #     self.codon_token_to_index.get(codon)
        #     for codon in self._split_into_triplets(self.parse_seq(seq))
        # ]
        tokens = [
            self.token_to_index.get(codon)
            for codon in self._split_into_triplets(self.parse_seq(seq))
        ]
        if add_special_tokens:
            tokens = (
                #[self.start_token_id]
                tokens
                + [self.end_token_id]
            )
        return tokens


    def _tokenise_nt_seqs(self, seqs):
        """
        This function performs tokenisation for a whole data set of
        nucleotide sequences.

        For this purpose, it employs the private function 
        `_tokenise_nt_seq`.

        Parameters
        ----------
        seqs: iterable, dtype=str or dtype=binary
            An iterable containing the nucleotide sequences to be tokenised.
        seq_len: int
            Integer indicating the maximum sequence length, but in terms of
            codons / triplets instead of in terms of single nucleotides.
            Input sequences shorter than this value are padded with a
            special token until they reach the specified length. When
            passing this value to the function call, it should be taken into
            consideration that the <START> and <END> tokens are already
            added to each individual sequence. Hence, in order to avoid a
            ValueError, the sequence length needs to be incremented by 2.

            Apart from that, the tokenised nucleotide sequence is shortened
            by one unit at the beginning and the end later on, respectively.
            Hence, its length must be greater by 1 than the length of the
            corresponding amino acid sequence. This, however, is handled by
            the function itself, so that the user can insert the same
            sequence length for both the amino acid and the nucleotide
            sequence.

        Returns
        -------
        tokenised_seqs_array: ndarray, dtype=int32, shape=(m, n)
            A NumPy array harbouring  the tokenised analogues of the input
            nucleotide sequences. The first dimension, m, corresponds to the
            amount of input sequences, whereas the second dimension, n,
            corresponds to the largest sequence length of the batch.
        """
        # First, perform the mere tokenisation
        tokenised_seqs_list = []

        for seq in seqs:
            tokenised_seqs_list.append(self._tokenise_nt_seq(seq))

        # Perform padding for each of the tokenised sequences
        # Determine the largest sequence length in the batch and increment
        # it by 1
        max_length = max(map(len, tokenised_seqs_list))
        max_length += 1

        tokenised_seqs_array = np.array(
            [tokenised_seq
            +
            (
                max_length - len(tokenised_seq)
            )*[self.pad_token_id]
            for tokenised_seq in tokenised_seqs_list],
            dtype=np.int32
        )

        #return tokenised_seqs_array
        return tokenised_seqs_list


    def tokenise_nt_seqs_for_eval(self, seqs, max_seq_len):
        """
        This function performs tokenisation for a whole data set of
        nucleotide sequences. For this purpose, it employs the private
        function '_tokenise_nt_seq'.

        The difference between this function and '_tokenise_nt_seqs' is that
        this function has an additional parameter, 'max_seq_len', indicating
        the maximum sequence length in terms of codons/triplets. This is
        opposed to the behaviour of the function '_tokenise_nt_seqs', which
        adopts the length of the largest input sequence as the maximum
        sequence length.
        
        Due to this implementation, 'max_seq_len' must be at least equal to
        the length of the largest input sequence. Otherwise, a ValueError is
        raised.

        Parameters
        ----------
        seqs: iterable, dtype=str or dtype=binary
            An iterable containing the nucleotide sequences to be tokenised.
        max_seq_len: int
            An integer indicating the maximum sequence length, but in terms
            of codons/triplets instead of in terms of single nucleotides.
            Input sequences shorter than this value are padded with a
            special token until they reach the specified length. When
            passing this value to the function call, it should be taken into
            consideration that the <START> and <END> tokens are already
            added to each individual sequence. Hence, in order to avoid a
            ValueError, the sequence length needs to be incremented by 2.

            Apart from that, the argument passed to this parameter needs to
            be at least equal to the length of the largest input sequence.
            Otherwise, a ValueError is raised.

        Returns
        -------
        tokenised_seqs_array: ndarray, dtype=int32, shape=(m, n)
            A NumPy array harbouring the tokenised analogues of the input
            nucleotide sequences. The first dimension, m, corresponds to the
            amount of input sequences, whereas the second dimension, n,
            corresponds to the maximum sequence length.
        """
        # First, check for a valid argument passed to 'max_seq_len'
        # Keep in mind that the parameter 'max_seq_len' refers to the
        # maximum sequence length in terms of codons, not in terms of single
        # nucleotides
        # Hence, in order to perform the comparison, the length of the
        # largest input nucleotide sequence must be divided by three
        max_input_length = max(map(len, seqs)) / 3
        if max_seq_len < max_input_length:
            raise ValueError(
                "The argument passed to 'max_seq_len' must be at least "
                "equal to the length of the largest input sequence!"
            )
        
        tokenised_seqs_list = []

        for seq in seqs:
            tokenised_seqs_list.append(self._tokenise_nt_seq(seq))
        
        # Perform padding for each of the tokenised sequences
        tokenised_seqs_array = np.array(
            [tokenised_seq
            +
            (
                max_seq_len - len(tokenised_seq)
            ) * [self.pad_token_id]
            for tokenised_seq in tokenised_seqs_list],
            dtype=np.int32
        )

        return tokenised_seqs_array

    ##########################
    ## ADDED BY RHIVU
    ##########################

    # def tokenise_aa_nt_pair(self, aa_seqs, nt_seqs):
    #     aa_tokens = self.tokenise_aa_seqs(aa_seqs)
    #     nt_tokens = self._tokenise_nt_seqs(nt_seqs
    #     )

    #     BOS = self.additional_aa_token_to_index["<START>"]
    #     SEP = self.additional_aa_token_to_index["<SEP>"]
    #     EOS = self.additional_aa_token_to_index["<END>"]

    #     return aa_tokens + [SEP] + nt_tokens

    def tokenise_aa_nt_pair(self, aa_seqs, nt_seqs):
        """
        Tokenises and concatenates AA and NT sequences with special tokens.

        Parameters:
        - aa_seqs: list of amino acid sequences (str)
        - nt_seqs: list of nucleotide sequences (str)

        Returns:
        - tokenised_pairs: Tensor of shape (batch_size, sequence_len)
        """

        assert len(aa_seqs) == len(nt_seqs), "AA and NT sequence batch sizes must match"

        # Tokenise each individually
        aa_tokens = self.tokenise_aa_seqs(aa_seqs)     # shape: (B, L_aa)
        #print(f"AA tokens: {aa_tokens}")
        nt_tokens = self._tokenise_nt_seqs(nt_seqs)    # shape: (B, L_nt)
        #print(f"NT tokens: {nt_tokens}")

        SEP = self.sep_token_id 
        batch_size = len(aa_seqs)

        combined_tokens = []

        for i in range(batch_size):
            aa_seq = aa_tokens[i]
            nt_seq = nt_tokens[i]

            combined = np.concatenate([
                aa_seq, 
                np.array([SEP], dtype=np.int32), 
                nt_seq
            ])
            combined_tokens.append(combined)
            #print(f"Combined tokens for sequence {i}: {combined}")

        # Pad to max length in batch
        # max_len = max(len(seq) for seq in combined_tokens)
        # PAD_ID = self.pad_token_id

        # padded_tokens = np.array([
        #     np.pad(seq, (0, max_len - len(seq)), constant_values=PAD_ID)
        #     for seq in combined_tokens
        # ], dtype=np.int32)
        #print(f"Padded tokens: {padded_tokens}")

        # return torch.tensor(padded_tokens, dtype=torch.long)
        return torch.tensor(combined_tokens, dtype=torch.long)

    def _detokenise_nt_seqs_torch(self, nt_token_tensor):
        """
        This function performs detokenisation for a whole data set of
        nucleotide sequences using PyTorch tensors.

        Parameters
        ----------
        nt_token_tensor: torch.Tensor, dtype=torch.int
            A tensor containing the sequences of token IDs generated in the
            course of translation.

        Returns
        -------
        nt_seq_array: NumPy array, dtype=str
            A one-dimensional NumPy array containing the sequences generated
            during translation as strings. In other words, it contains the
            generated sequences as human-readable nucleotide sequences.
        """
        # Convert the PyTorch tensor into a NumPy array for easier handling
        nt_token_array = nt_token_tensor.cpu().numpy()

        nt_seq_list = []

        for token_seq in nt_token_array:
            # Map each token ID to its corresponding codon using the dictionary
            codon_list = [
                self.index_to_codon_token.get(token)
                for token in token_seq
            ]
            # Join the codons into a single nucleotide sequence string
            codon_str = "".join(codon_list)
            nt_seq_list.append(codon_str)

        # Convert the list into a NumPy array
        nt_seq_array = np.array(nt_seq_list, dtype=str)

        return nt_seq_array

    def _detokenise_nt_seqs_torch(self, nt_token_tensor):
        """
        Converts token IDs to nucleotide sequences, stripping <END> and <PAD> tokens.

        Args:
            nt_token_tensor (List[List[int]] | Tensor | np.ndarray)

        Returns:
            List[str]: List of nucleotide sequences (e.g. ['ATGGCC...', ...])
        """
        if isinstance(nt_token_tensor, torch.Tensor):
            nt_token_tensor = nt_token_tensor.detach().cpu().tolist()
        elif isinstance(nt_token_tensor, np.ndarray):
            nt_token_tensor = nt_token_tensor.tolist()

        sequences = []
        for row in nt_token_tensor:
            codons = []
            for token_id in row:
                if token_id == self.end_token_id or token_id == self.pad_token_id:
                    break
                codon = self.index_to_token.get(token_id, '')  # handle unknown token_ids
                codons.append(codon)
            sequences.append(''.join(codons))

        return sequences


    #########################
    ## ADDED BY RHIVU
    #########################
    ## BELOW IS THE FUNCTION FOR PYTORCH
    def format_dataset_torch(self, aa_nt_pairs, max_seq_length, device='cuda'):
        #print(f"Formatting dataset with max_seq_length={max_seq_length} on device={device}")
        aa_nt_pairs = aa_nt_pairs.to_numpy()
        aa_seqs, nt_seqs = zip(*aa_nt_pairs)

        input_id_list = []
        label_list = []
        pad_id = self.pad_token_id  

        for aa, nt in zip(aa_seqs, nt_seqs):
            #print(f"Processing sequence: {aa} -> {nt}")
            # Tokenise amino acid and nucleotide sequences
            token_ids = self.tokenise_aa_nt_pair([aa], [nt])
            #print(f"Token IDs: {token_ids}")
            token_ids = token_ids.flatten().tolist()
            #print(f"Flattened Token IDs: {token_ids}, with length {len(token_ids)}")

            if self.sep_token_id not in token_ids:
                #print("No <SEP> token found in sequence, skipping.")
                continue  # malformed sequence

            sep_idx = token_ids.index(self.sep_token_id)
            #print(f"SEP token found at index: {sep_idx}")

            labels = []
            for i, tid in enumerate(token_ids):
                if i <= sep_idx:
                    labels.append(-100)  # Mask AA + SEP
                else:
                    labels.append(tid)   # Predict codons
            
            input_id_list.append(token_ids)
            label_list.append(labels)

            # # Pad or truncate
            # if len(token_ids) > max_seq_length:
            #     token_ids = token_ids[:max_seq_length]
            #     labels = labels[:max_seq_length]
            # else:
            #     token_ids += [pad_id] * (max_seq_length - len(token_ids))
            #     labels += [-100] * (max_seq_length - len(labels))

            # # ⚠️ Skip if all labels are -100
            # if all(l == -100 for l in labels):
            #     continue
            


            assert len(input_id_list) == len(label_list), "Input and label lists must be of the same length."
            #print(f"input id list: {input_id_list}")
            #print(f"label list: {label_list}")

        if len(input_id_list) == 0:
            # make sure this is a list-of-list, not a flat list!
            input_id_list = [ input_id_list[0] ]    # already a list, so this is [[...]]
            label_list    = [ label_list[0] ]

        input_ids = torch.tensor(input_id_list, dtype=torch.long, device=device)
        labels = torch.tensor(label_list, dtype=torch.long, device=device)
        attention_mask = (input_ids != pad_id).long()

        #print("→ [format_dataset_torch] input_ids.shape :", input_ids.shape)
        #print("→ [format_dataset_torch] attention_mask.shape :", attention_mask.shape)
        #print("→ [format_dataset_torch] labels.shape :", labels.shape)


        return input_ids, attention_mask, labels

    # def translate_aa_into_nt_torch(self, transformer, aa_seqs, max_seq_length, 
    #                                 return_string=True, batch_size=32, device='cpu'):
    #     """
    #     Translate amino acid sequences to nucleotide sequences using a PyTorch transformer.
    #     """
    #     aa_encoding = self.tokenise_aa_seqs(aa_seqs)
    #     aa_encoding = [torch.tensor(seq, dtype=torch.long) for seq in aa_encoding]
    #     aa_encoding = pad_sequence(aa_encoding, batch_first=True, padding_value=self.pad_token_id)
    #     aa_encoding = aa_encoding.to(device)

    #     n_batches = ceil(len(aa_encoding) / batch_size)

    #     nt_translations = []

    #     start_token = self.start_token_id
    #     end_token = self.end_token_id
    #     pad_token = self.pad_token_id

    #     transformer.eval()
    #     with torch.no_grad():
    #         for i in range(n_batches):
    #             batch_aa_encoding = aa_encoding[i*batch_size:(i+1)*batch_size]
    #             batch_size_curr = batch_aa_encoding.size(0)

    #             # Initial decoder input: <START>
    #             generated = torch.full((batch_size_curr, 1), start_token, dtype=torch.long, device=device)

    #             for _ in range(max_seq_length):
    #                 # Concatenate AA input and generated NTs
    #                 input_ids = torch.cat([batch_aa_encoding, generated], dim=1)

    #                 # Padding mask: True where input is NOT pad
    #                 attention_mask = (input_ids != pad_token)

    #                 # Forward pass through model
    #                 output = transformer(input_ids, attention_mask=attention_mask)
    #                 output_logits = output.logits

    #                 next_token_logits = output_logits[:, -1, :]  # get last token's logits
    #                 next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    #                 # # assert that we never predict an AA id (> 4 and ≤ vocab_start_of_codons)
    #                 # aa_vocab_max = tokenizer.token_to_index['*']   # that’s 2 for <END>, but AA tokens start at 5
    #                 # first_codon_id = aa_vocab_max + 1
    #                 # # sanity‐check during decoding:
    #                 # bad = (next_tokens < first_codon_id) & (next_tokens != start_token) & (next_tokens != end_token)
    #                 # mask = next_tokens >= first_codon_id
    #                 # assert mask.all(), "❌ Model predicted an AA or special token after <SEP>!"

    #                 generated = torch.cat([generated, next_tokens], dim=1)

    #                 if (next_tokens == end_token).all():
    #                     break

    #             # Remove <START> token and truncate at <END>
    #             generated_np = generated[:, 1:].cpu().numpy()
                
    #             if return_string:
    #                 batch_translations = self._detokenise_nt_seqs(generated_np)
    #                 nt_translations.extend(batch_translations)
    #             else:
    #                 nt_translations.append(generated_np)

    #     if not return_string:
    #         nt_translations = np.vstack(nt_translations)

    #     return np.array(nt_translations)
    def translate_aa_into_nt_torch(self, transformer, aa_seqs, max_seq_length,
                                    return_string=True, batch_size=32, device='cpu', temperature=1.0):
        """
        Translate amino acid sequences to nucleotide sequences using a PyTorch transformer.
        Restricted sampling: only codon token IDs are allowed when sampling.
        """
        aa_encoding = self.tokenise_aa_seqs(aa_seqs)
        aa_encoding = [torch.tensor(seq, dtype=torch.long) for seq in aa_encoding]
        aa_encoding = pad_sequence(
            aa_encoding,
            batch_first=True,
            padding_value=self.pad_token_id,
        ).to(device)

        # Append a <SEP> token so inputs are [<START>, aa…, <SEP>] prior to generation
        sep_tokens = torch.full(
            (aa_encoding.size(0), 1),
            self.sep_token_id,
            dtype=torch.long,
            device=device,
        )
        aa_encoding = torch.cat([aa_encoding, sep_tokens], dim=1)

        n_batches = ceil(len(aa_encoding) / batch_size)
        nt_translations = []

        end_token   = self.end_token_id
        pad_token   = self.pad_token_id

        # where AA/vocab ends and codon IDs begin
        aa_vocab_max   = self.token_to_index['*']      # last AA or special
        first_codon_id = aa_vocab_max + 1             # first codon token ID

        transformer.eval()
        with torch.no_grad():
            for i in range(n_batches):
                aa_with_sep = aa_encoding[i * batch_size : (i + 1) * batch_size]
                batch_size_curr = aa_with_sep.size(0)

                # start with an empty generated sequence
                generated = torch.empty(
                    batch_size_curr, 0, dtype=torch.long, device=device
                )

                # mask tracking which sequences have hit <END>
                done = torch.zeros(batch_size_curr, dtype=torch.bool, device=device)

                for _ in range(max_seq_length):
                    # build the full input and mask
                    input_ids = torch.cat([aa_with_sep, generated], dim=1)
                    attention_mask = (input_ids != pad_token)

                    # forward pass through provided transformer-like callable
                    output = transformer(input_ids, attention_mask=attention_mask)
                    # wrapper returns SimpleNamespace-like object with .logits
                    output_logits = output.logits  # (B, L, V)
                    next_token_logits = output_logits[:, -1, :].clone()  # (B, V)

                    # ---- Restrict sampling to codon token ids only ----
                    # disallow prediction of any token <= aa_vocab_max (i.e. AA tokens / specials)
                    if first_codon_id > 0:
                        next_token_logits[:, :first_codon_id] = -1e9

                    # If any sequence is already done, force it to predict END token only
                    if done.any():
                        # set all logits to -inf for done rows, except end_token
                        mask_done = done.nonzero(as_tuple=True)[0]
                        if mask_done.numel() > 0:
                            next_token_logits[mask_done, :] = -1e9
                            next_token_logits[mask_done, end_token] = 0.0  # keep end_token highest

                    # sampling with temperature
                    scores = next_token_logits / max(1e-8, temperature)
                    probs = torch.softmax(scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)  # (B,1)

                    # update done mask
                    next_tokens_flat = next_tokens.squeeze(1)
                    done = done | (next_tokens_flat == end_token)

                    # append new tokens
                    generated = torch.cat([generated, next_tokens], dim=1)

                    # break if all finished
                    if done.all():
                        break

                # convert generated tokens to numpy
                generated_np = generated.cpu().numpy()

                if return_string:
                    batch_translations = self._detokenise_nt_seqs(generated_np)
                    nt_translations.extend(batch_translations)
                else:
                    nt_translations.append(generated_np)

        if not return_string:
            nt_translations = np.vstack(nt_translations)

        return np.array(nt_translations)


    def _detokenise_nt_seqs(self, nt_token_array):
        """
        Converts a batch of tokenised codon sequences (as NumPy array) back into nucleotide strings.
        """
        nt_seqs = []
        for seq in nt_token_array:
            codons = []
            for token_id in seq:
                if token_id == self.end_token_id:
                    break
                if token_id in [self.start_token_id, self.pad_token_id]:
                    continue
                codon = self.index_to_token.get(token_id, 'NNN')
                codons.append(codon)
            nt_seqs.append("".join(codons))
        return nt_seqs



    def _detokenise_aa_seqs(self, aa_id_seqs):
        """
        Convert a list of amino acid token ID sequences into human-readable strings,
        removing special tokens like <PAD>, <SEP>, <START>, and <END>.

        Args:
            aa_id_seqs (List[List[int]]): A list where each element is a list of amino acid token IDs.

        Returns:
            List[str]: Decoded amino acid sequences as strings.
        """
        aa_seqs = []

        # Define set of special tokens to ignore
        special_token_ids = {
            self.pad_token_id,
            self.sep_token_id,
            self.start_token_id,
            self.end_token_id
        }

        for id_seq in aa_id_seqs:
            # Convert to NumPy array if it's a PyTorch tensor
            if isinstance(id_seq, torch.Tensor):
                id_seq = id_seq.cpu().numpy()

            aa_chars = []
            for token_id in id_seq:
                if token_id in special_token_ids:
                    continue  # skip special tokens entirely
                aa = self.index_to_token.get(token_id, '')  # fallback to '' if unknown
                aa_chars.append(aa)

            aa_seq = ''.join(aa_chars)
            aa_seqs.append(aa_seq)

        return aa_seqs


    def compute_levenshtein_distances(self, predictions, references, normalize=True):
        """
        Compute Levenshtein distances (raw or normalised) between predicted and reference sequences.

        Args:
            predictions (list[str]): List of predicted nucleotide sequences.
            references (list[str]): List of ground truth nucleotide sequences.
            normalize (bool): If True, return normalised distances (0–1). Else, return raw distances.

        Returns:
            list[float]: List of distances (float if normalised, int otherwise)
        """
        distances = []
        for pred, ref in zip(predictions, references):
            if len(ref) == 0:
                distances.append(1.0 if normalize else len(pred))  # Fallback for empty reference
                continue

            dist = Levenshtein.distance(pred, ref)
            if normalize:
                dist = dist / max(len(pred), len(ref))

            distances.append(dist)

        return distances

    ##############################
    ## ADDED BY RHIVU
    ##############################
    # def generate_biotokenizer_aa_json(self, output_file="biotokenizer_aa.json"):
    #     """
    #     This function generates a biotokenizer JSON file for amino acid vocabulary.
    #     It creates a mapping of ALL_AAS and ADDITIONAL_TOKENS to their unique indices.

    #     Parameters
    #     ----------
    #     output_file: str
    #         The name of the JSON file to save the vocabulary mapping.
    #     """
    #     # Combine ALL_AAS and ADDITIONAL_TOKENS into a single vocabulary
    #     vocab = self.ADDITIONAL_TOKENS + list(self.ALL_AAS)

    #     # Create a mapping of tokens to unique indices
    #     vocab_mapping = {token: idx for idx, token in enumerate(vocab)}

    #     # Save the mapping as a JSON file
    #     with open(output_file, "w") as f:
    #         json.dump(vocab_mapping, f, indent=4)

    #     print(f"Biotokenizer JSON file saved to {output_file}")

    # ##############################
    # ## ADDED BY RHIVU   
    # ##############################

    # def generate_biotokenizer_codon_json(self, output_file="biotokenizer_codon.json"):
    #     """
    #     This function generates a biotokenizer JSON file for codon vocabulary.
    #     It creates a mapping of ALL_CODONS and ADDITIONAL_NT_TOKENS to their unique indices.

    #     Parameters
    #     ----------
    #     output_file: str
    #         The name of the JSON file to save the vocabulary mapping.
    #     """
    #     # Combine ALL_CODONS and ADDITIONAL_NT_TOKENS into a single vocabulary
    #     vocab = self.ADDITIONAL_NT_TOKENS + self.ALL_CODONS

    #     # Create a mapping of tokens to unique indices
    #     vocab_mapping = {token: idx for idx, token in enumerate(vocab)}

    #     # Save the mapping as a JSON file
    #     with open(output_file, "w") as f:
    #         json.dump(vocab_mapping, f, indent=4)

    #     print(f"Biotokenizer JSON file saved to {output_file}")

    '''
    def translate_aa_into_nt(
        transformer, aa_seqs, max_seq_length, return_string=True, batch_size=32
    ):
        """
        This function translates from the "language" of amino acids into one
        specific "dialect" of the nucleotide sequence "language". The term
        "dialect" denotes the codon usage bias of one specific organism,
        e. g. E. coli or S. cerevisiae. In order to accomplish this task,
        the function accepts as input a transformer neural network which was
        trained to translate amino acid sequences into nucleotide sequences
        of one specific organism.

        For this purpose, it employs the private functions `_encode_aa_seqs`
        as well as `_detokenise_nt_seqs`.

        In order to circumvent an OOM error in case of very large test data
        sets, the test data set is manually batched into batches of a
        user-defined size; the default value is 32.

        Parameters
        ----------
        transformer: Keras model
            A transformer neural network implemented in Keras and performing
            the translation of amino acid sequences into nucleotide
            sequences according to the codon usage bias of one specific
            organism.
        aa_seqs: list
            A list comprising the amino acid sequences to be translated into
            nucleotide sequences in accordance with the codon usage bias of
            the respective organism.
        max_seq_length: int
            An integer denoting the maximum sequence length after which to
            stop translation in case that the <END> token is not produced.
        return_string: boolean, optional
            A boolean indicating whether the translations are supposed to be
            returned as contiguous strings (i. e. one string for each input
            amino acid sequence) or as integer tokens. Defaults to 'True'.
        batch_size: int
            An integer defining the batch size. The segmentation of the data
            set into batches of a defined size happens in order to avoid an
            OOM error.

        Returns
        -------
        nt_translations: NumPy array, dtype=str or NumPy array, dtype=int
            Depending on the argument of the optional parameter
            'return_string', two different outputs can be returned. If the
            optional parameter is set to 'True', a one-dimensional NumPy
            array is returned harbouring the sequences generated during
            translation as strings. In other words, it contains the
            generated sequences as human-readable nucleotide sequences. Its
            length equals the amount of amino acid sequences passed as
            input, i. e. the length of `aa_seqs`. However, if the optional
            parameter is set to 'False', a two-dimensional NumPy array is
            returned harbouring the sequences generated during translation
            as integer tokens. Each row of this two-dimensional array
            corresponds to one translation.
        """
        # Tokenise the input amino acid sequences
        aa_encoding = tokenise_aa_seqs(aa_seqs)

        # The manual batching of the tokenised amino acid sequences is
        # performed
        # To this end, the total amount of batches having the size defined
        # by the user is determined
        n_batches = ceil(len(aa_encoding) / batch_size)

        # As stated in the documentation string, the output depends on the
        # value of the optional parameter 'return_string'
        # Hence, depending on whether the parameter is set to 'True' or
        # 'False', a different output array is created
        if return_string:
            nt_translations = np.array([], dtype=str)
        else:
            nt_translations = np.empty(
                shape=(0, max_seq_length), dtype=np.int32
            )

        for i in range(n_batches):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size

            # Extract the batch currently dealt with
            current_batch_aa_encoding = aa_encoding[start_index:end_index, :]

            # Define a function that computes the token probabilities for the
            # next position given the input sequence
            # The transformer output has the shape (B, l, 68), where B denotes
            # the batch size/amount of amino acid sequences entered as input,
            # l denotes the maximum sequence length and 68 is the total amount
            # of nucleotide tokens
            # As only the probabilities of the next token to predict are wanted,
            # "-1" is chosen in the second dimension during slicing
            def token_probability_fn(decoder_input_tokens):
                return transformer(
                    # The keyword argument `training` is set to `False` as this
                    # makes inference more efficient by only computing the last
                    # the last prediction instead of the prediction for all
                    # tokens
                    # It is not necessary to distinguish between training and
                    # inference as this function is exclusively used for
                    # inference; hence, this keyword argument does not need to
                    # be accessible to the user
                    [current_batch_aa_encoding, decoder_input_tokens],
                    training=False
                )[:, -1, :]
            
            n_seqs = len(current_batch_aa_encoding)

            # Initialise the translated nucleotide sequences with the "<START>"
            # token
            prompt = tf.fill(
                (n_seqs, 1),
                global_codon_token_to_index["<START>"]
            )

            # Strangely enough, it is stated in the Keras documentation that
            # 'keras_nlp.utils.greedy_search' returns either a 1D int Tensor
            # or a 2D int RaggedTensor, whereby a ragged tensor is a
            # non-rectangular tensor whose entries, i. e. rows have
            # different sizes
            # However, this is not true as the chosen padding token is added
            # after encountering the end token until the maximum sequence
            # length is reached
            generated_tokens = keras_nlp.utils.greedy_search(
                token_probability_fn,
                prompt,
                max_length=max_seq_length,
                end_token_id=global_codon_token_to_index["<END>"]
            )

            if return_string:
                current_batch_nt_translations = _detokenise_nt_seqs(generated_tokens)
                nt_translations = np.append(nt_translations, current_batch_nt_translations)
            else:
                current_batch_nt_translations = generated_tokens.numpy()
                # Verify that the array harbouring the translations is two-
                # dimensional; an one-dimensional array is returned when the
                # batch size is 1, which can occur under certain
                # circumstances
                if current_batch_nt_translations.ndim < 2:
                    current_batch_nt_translations = np.expand_dims(
                        current_batch_nt_translations, axis=0
                    )
                # Now, append the new translations to the output array
                nt_translations = np.append(
                    nt_translations, current_batch_nt_translations, axis=0
                )

        return nt_translations
    '''