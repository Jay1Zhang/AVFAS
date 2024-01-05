from collections import namedtuple

Genotype = namedtuple('Genotype', ['amt', 'vmt', 'avmt'])
FusionGenotype = namedtuple('FusionGenotype', ['edges', 'ops'])
CrossFusionGenotype = namedtuple('CrossFusionGenotype', ['edges', 'poses', 'wins', 'ops'])
