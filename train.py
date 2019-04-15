from emode_hdnnp import network

def main():
	net = network.EMHDNNP(mol_set_name="master_jeherr2_HCNOFPSClSeBrI.pkl")
	net.start_training()

if __name__ == "__main__":
	main()
