from diffwofost.physical_models.parameter_providers import ParameterProvider


class TestParameterProvider:
    def test_parameter_provider_supports_dict_methods(self):
        p = ParameterProvider(
            sitedata={"A": 0}, timerdata={"B": 1}, soildata={"C": 2}, cropdata={"D": 3}
        )
        p.set_override("E", 4, check=False)
        assert len(p.items()) == len(p.keys()) == len(p.values()) == 5
        assert set(p.keys()) == set("ABCDE")
        assert set(p.values()) == set(range(5))
