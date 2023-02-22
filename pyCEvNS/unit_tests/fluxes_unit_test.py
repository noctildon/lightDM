from unittest import TestCase, main



from pyCEvNS.flux import NeutrinoFluxFactory
# neutrino flux factory
class NeutrinoTestCase(TestCase):
    def test_flux_factory(self):
        # check they all load properly
        nuff = NeutrinoFluxFactory()
        for name in nuff.flux_list:
            nuff.get(name, zenith=0.025)
    
    def test_neutrino_flux(self):
        coh = NeutrinoFluxFactory().get("coherent")
        assert(coh.integrate(0, 200, "mu") > 0)



# dm flux factory



# axion flux factory








if __name__ == "__main__":
    main()