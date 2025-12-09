from argparse import ArgumentParser

class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		
		self.parser.add_argument('--batch_size', default=1, type=int, help='Batch size for inference')
		self.parser.add_argument('--workers', default=4, type=int, help='Number of test dataloader workers')
		
		self.parser.add_argument('--stylegan_weights', default='models/pretrained_models/stylegan2-ffhq-config-f.pt', type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--stylegan_size', default=1024, type=int)
		
		self.parser.add_argument("--threshold", type=int, default=0.03)
		self.parser.add_argument("--checkpoint_path", type=str, default='checkpoints/net_face.pth')
		self.parser.add_argument("--save_dir", type=str, default='output')
		self.parser.add_argument("--num_all", type=int, default=200)
		
		self.parser.add_argument("--target", type=str, required=True, help='Specify the target attributes to be edited')

		self.parser.add_argument("--test_latent_dir", type=str, default='./latent_code')
		#白色衣服的实验用white_classname，平均衣服的实验用images
		self.parser.add_argument('--white_classname', type=str, default='ffhq', help="sample cloths which turned to white")
		self.parser.add_argument('--images_classname', type=str, default='ffhq', help="sample cloths ")

		self.parser.add_argument('--texture_classname', type=str, default='ffhq', help="texture cropped from sample cloths ")
		self.parser.add_argument('--sampled_texture_classname', type=str, default='ffhq', help="texture cropped from sample cloths ")
		self.parser.add_argument("--sample", type=bool, default=False,help='is the test data sampled?')
		self.parser.add_argument('--modi_loss', default=False, type=bool, help='if divide fine and medium when calculate loss')
		self.parser.add_argument('--only_medium', default=False, type=bool, help='')

	def parse(self):
		opts = self.parser.parse_args()
		return opts