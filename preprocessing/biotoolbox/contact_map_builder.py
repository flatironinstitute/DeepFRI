import numpy as np
from Bio import Align
from Bio.Data.SCOPData import protein_letters_3to1
from Bio.SeqUtils import seq1

TEN_ANGSTROMS     = 10.0
ALIGNED_BY_SEQRES = 'aligned by SEQRES'
ATOMS_ONLY        = 'ATOM lines only'
INCOMPARABLE_PAIR = 10000.
KEY_NOT_FOUND     = 1000.

class ContactMapContainer:
    def __init__(self):
        self.chains = {}

    def with_chain(self, chain_name):
        self.chains[chain_name] = {}

    def with_chain_seq(self, chain_name, seq):
        self.chains[chain_name]['seq'] = seq

    def with_map_for_chain(self, chain_name, contact_map):
        self.chains[chain_name]['contact-map'] = contact_map

    def with_alignment_for_chain(self, chain_name, alignment):
        self.chains[chain_name]['alignment'] = alignment

    def with_method_for_chain(self, chain_name, method):
        self.chains[chain_name]['method'] = method

    def with_final_seq_for_chain(self, chain_name, final_seq):
        self.chains[chain_name]['final-seq'] = final_seq


def correct_residue(x, target):
    try:
        sl = protein_letters_3to1[x.resname]
        if sl == target:
            return True
        return False
    except KeyError:
        return False


class DistanceMapBuilder:
    def __init__(self,
                 atom="CA",
                 verbose=True,
                 pedantic=True,
                 glycine_hack=-1):

        self.verbose = verbose
        self.pedantic = pedantic
        self.set_atom(atom)
        if not isinstance(glycine_hack, (int, float)):
            raise ValueError(f"{glycine_hack} is not an int")
        self.glycine_hack = glycine_hack

    def speak(self, *args, **kwargs):
        """
        Print a message or blackhole it
        """
        if self.verbose:
            print(*args, **kwargs)

    def set_atom(self, atom):
        if atom.casefold() not in ['ca', 'cb']:
            raise ValueError(f"{atom.casefold()} not 'ca' or 'cb'")
        self.__atom = atom.upper()
        return self

    @property
    def atom(self):
        return self.__atom

    def generate_map_for_pdb(self, structure_container):

        aligner      = Align.PairwiseAligner()
        contact_maps = ContactMapContainer()
        model        = structure_container.structure[0]

        for chain_name in structure_container.chains:
            chain = structure_container.chains[chain_name]
            contact_maps.with_chain(chain_name)
            self.speak(f"\nProcessing chain {chain_name}")

            if chain['seqres-seq'] is not None and len(chain['seqres-seq']) > 0:
                contact_maps.with_method_for_chain(chain_name, ALIGNED_BY_SEQRES)
                seqres_seq = chain['seqres-seq']
                atom_seq   = chain['atom-seq']

                alignment = aligner.align(seqres_seq, atom_seq)
                specific_alignment = next(alignment)
                self.speak(f"Seqres seq: {seqres_seq}",
                           f"Atom seq:   {atom_seq}",
                           specific_alignment, sep='\n')

                contact_maps.with_alignment_for_chain(chain_name, specific_alignment)

                # It's actually much easier to just have biopython generate the string alignment
                # and use that as a guide.
                pattern = specific_alignment.__str__().split("\n")
                aligned_seqres_seq, mask, aligned_atom_seq = pattern[:3]

                # Build a list of residues that we do have atoms for.
                residues           = model[chain_name].get_residues()
                reindexed_residues = list(model[chain_name].get_residues())
                final_residue_list = []

                picked_residues = 0
                non_canonicals_or_het = 0

                for i in range(len(aligned_atom_seq)):
                    if aligned_seqres_seq[i] == '-':
                        # This is an inserted residue from the aligner that doesn't actually match any
                        # seqres line. Don't even insert a None.
                        continue
                    current_aligned_atom_residue_letter = aligned_atom_seq[i]
                    #  atom seq has a letter and the mask shows it corresponds to a seqres item
                    if current_aligned_atom_residue_letter != '-' and mask[i] == '|':
                        candidate_residue = next((x for x in reindexed_residues[picked_residues:picked_residues + 5] if
                                                  correct_residue(x, current_aligned_atom_residue_letter)), None)

                        if candidate_residue is None:
                            # The right answer is probably 'None' but we need to know why.
                            residue = reindexed_residues[picked_residues]
                            if residue.id[0].startswith('H_'):
                                non_canonicals_or_het += 1
                        else:
                            picked_residues += 1

                        final_residue_list.append(candidate_residue)
                    else:
                        final_residue_list.append(None)

                final_seq_three_letter_codes = ''.join(
                    [r.resname if r is not None else 'XXX' for r in final_residue_list])
                final_seq_one_letter_codes = seq1(final_seq_three_letter_codes, undef_code='-',
                                                  custom_map=protein_letters_3to1)
                self.speak(f"Final [len of seq {len(seqres_seq)}] [len of result {len(final_seq_one_letter_codes)}] "
                           f"[len of final residue list {len(final_residue_list)}]:\n{final_seq_one_letter_codes}")

                if self.pedantic and len(final_residue_list) != len(seqres_seq):
                    raise Exception(
                        f"Somehow the final residue list {len(final_residue_list)} doesn't match the size of the SEQRES seq {len(seqres_seq)}")

                if self.pedantic and (len(seqres_seq) != len(final_seq_one_letter_codes) != len(final_residue_list)):
                    raise Exception(
                        'The length of the SEQRES seq != length of final_seq_one_letter_codes != length of final residue list')

                sanity_check = aligned_atom_seq.replace('X', '')
                if self.pedantic and sanity_check != final_seq_one_letter_codes:
                    print(f"sanity_check {sanity_check}")
                    print(f"final_seq    {final_seq_one_letter_codes}")
                    count = sum(1 for a, b in zip(sanity_check, final_seq_one_letter_codes) if a != b)
                    # While going through the data we found some _very_ large structures in the PDB.
                    # Some of them have massive interior chains w/ tons of missing data. In this case
                    # we're basically just saying we did what we could, passing the data along and saying
                    # we still were in pedantic mode.
                    missing_residue_heuristic = sanity_check.count('-') / len(sanity_check)
                    missing_residue_heuristic_2 = final_seq_one_letter_codes.count('-') / len(final_seq_one_letter_codes)
                    if count == non_canonicals_or_het:
                        # Add a message about this.
                        print(
                            f"Warning: The final sequence and the sanity check were different, but the difference equals the number of HETATMs or non-canonical residues. _Probably_ OK.")
                    elif missing_residue_heuristic >= 0.5 or missing_residue_heuristic_2 >= 0.5:
                        print(f"Warning: The final sequence and the sanity check were different. Over 50% of the chain is unresolved. Nothing we can do about it.")
                    else:
                        print ("Vlada")
                        # raise Exception(
                        #    f'The final one letter SEQ generated from residues does not match the aligned atom seq (Diff count {count} but HETATM {non_canonicals_or_het})')

                contact_maps.with_final_seq_for_chain(chain_name, final_seq_one_letter_codes)
                contact_maps.with_chain_seq(chain_name, seqres_seq)
                contact_maps.with_map_for_chain(chain_name,
                                                self.__residue_list_to_contact_map(final_residue_list, len(seqres_seq)))
            else:
                contact_maps.with_method_for_chain(chain_name, ATOMS_ONLY)
                atom_seq = chain['atom-seq']
                residues = model[chain_name].get_residues()

                final_residue_list = []
                missing_alpha_carbons = []
                for r in residues:
                    try:
                        _ = r["CA"]
                        final_residue_list.append(r)
                    except KeyError:
                        missing_alpha_carbons.append(r)

                # Sanity checks
                final_seq_three_letter_codes = ''.join(
                    [r.resname if r is not None else 'XXX' for r in final_residue_list])
                final_seq_one_letter_codes = seq1(final_seq_three_letter_codes, undef_code='-',
                                                  custom_map=protein_letters_3to1)
                print(final_seq_one_letter_codes)
                corrected_atom_seq = final_seq_one_letter_codes
                # End sanity checks

                contact_maps.with_chain_seq(chain_name, corrected_atom_seq)
                contact_maps.with_map_for_chain(chain_name,
                                                self.__residue_list_to_contact_map(final_residue_list, len(corrected_atom_seq)))

        return contact_maps

    def __residue_list_to_contact_map(self, residue_list, length):
        dist_matrix = self.__calc_dist_matrix(residue_list)
        diag = self.__diagnolize_to_fill_gaps(dist_matrix, length)
        #contact_map = self.__create_adj(diag, TEN_ANGSTROMS)
        contact_map = diag
        return contact_map

    def __norm_adj(self, A):
        #  Normalize adj matrix.
        with np.errstate(divide='ignore'):
            d = 1.0 / np.sqrt(A.sum(axis=1))
        d[np.isinf(d)] = 0.0

        # normalize adjacency matrices
        d = np.diag(d)
        A = d.dot(A.dot(d))

        return A

    def __create_adj(self, _A, thresh):
        # Create CMAP from distance
        A = _A.copy()
        with np.errstate(invalid='ignore'):
            A[A <= thresh] = 1.0
            A[A > thresh] = 0.0
            A[np.isnan(A)] = 0.0
            A = self.__norm_adj(A)

        return A

    def __calc_residue_dist(self, residue_one, residue_two):
        """Returns the `self.atom` distance between two residues"""
        if bool({residue_one, residue_two} & {None}):
            return INCOMPARABLE_PAIR
        try:
            dist = self.__euclidean(residue_one, self.atom,
                                    residue_two, self.atom)
        except KeyError:
            if self.atom == "CB":
                if self.glycine_hack < 0: # CA-mode for CB+GLY
                    try:
                        dist = self.__euclidean(residue_one,'CA',
                                                residue_two,'CA')
                    except KeyError:
                        dist = KEY_NOT_FOUND
                else:
                    dist = self.glycine_hack
            else:
                dist = KEY_NOT_FOUND
        return dist

    def __euclidean(self, res1, atom1, res2, atom2):
        diff = res1[atom1] - res2[atom2]
        return np.sqrt(np.sum(diff * diff))


    def __diagnolize_to_fill_gaps(self, distance_matrix, length):
        # Create CMAP from distance
        A = distance_matrix.copy()
        for i in range(length):
            if A[i][i] == INCOMPARABLE_PAIR:
                A[i][i] = 1.0
                try:
                    A[i + 1][i] = 1.0
                except IndexError:
                    pass
                try:
                    A[i][i + 1] = 1.0
                except IndexError:
                    pass

        return A

    def __calc_dist_matrix(self, chain_one):
        """Returns a matrix of C-alpha distances between two chains"""
        answer = np.zeros((len(chain_one), len(chain_one)), np.float)
        for row, residue_one in enumerate(chain_one):
            for col, residue_two in enumerate(chain_one[row:], start=row):
                if col >= len(chain_one):
                    continue  # enumerate syntax is convenient, but results in invalid indices on last column
                answer[row, col] = self.__calc_residue_dist(residue_one, residue_two)
                answer[col, row] = answer[row, col]  # cchandler
        return answer
