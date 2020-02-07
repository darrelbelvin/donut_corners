from dc_tests import test_building

for i in range(1,10):
    test_building(i, None, score_all=True, save_prefix = f'figures/bldg-{i}')
    print(f"completed {i}")
