import torch
import numpy as np
import os
import random
import torch.nn.functional as F

from utils.utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy, verify_checkpoint_dir, task_confusion
from torch.optim import lr_scheduler
from video_reader import VideoDataset
from torch.utils.tensorboard import SummaryWriter

def getWIFN(seed):
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
    return worker_init_fn

class Learner:
    def __init__(self, cfg):
        self.cfg = self.parse_config(cfg)

        self.checkpoint_dir, self.logfile, self.test_checkpoint_path, self.resume_checkpoint_path \
            = get_log_files(cfg)

        print_and_log(self.logfile, "Options: %s\n" % self.cfg)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)
        if cfg.TEST.ONLY_TEST:
            print_and_log(self.logfile, "ONLY_TEST::Checkpoint Path: %s\n" % self.test_checkpoint_path)

        self.train_episodes = cfg.TRAIN.TRAIN_EPISODES
        self.test_episodes = cfg.TEST.TEST_EPISODES

        #self.writer = SummaryWriter()
        mode = 'test' if cfg.TEST.ONLY_TEST else 'train'
        ######################################################################################
        self.writer = SummaryWriter(comment=f"=>{cfg.MODEL.NAME}_{mode}_{cfg.DATA.DATASET}::{cfg.MODEL.BACKBONE}_{cfg.TRAIN.WAY}-{cfg.TRAIN.SHOT}_{cfg.TRAIN.QUERY_PER_CLASS}",flush_secs = 30)
        ######################################################################################
        
        #gpu_device = 'cuda:0'
        gpu_device = cfg.DEVICE.DEVICE
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        
        print("Random Seed: ", cfg.MODEL.SEED)
        np.random.seed(cfg.MODEL.SEED)
        random.seed(cfg.MODEL.SEED)
        torch.manual_seed(cfg.MODEL.SEED)
        torch.cuda.manual_seed(cfg.MODEL.SEED)
        torch.cuda.manual_seed_all(cfg.MODEL.SEED)
        torch.backends.cudnn.deterministic = True

        self.model = self.init_model()
        self.vd = VideoDataset(self.cfg)
        self.video_loader = torch.utils.data.DataLoader(self.vd, batch_size=1, num_workers=self.cfg.DATA.NUM_WORKERS, worker_init_fn=getWIFN(cfg.MODEL.SEED))
        
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        
        if self.cfg.SOLVER.OPTIM_METHOD == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.SOLVER.LR)
        elif self.cfg.SOLVER.OPTIM_METHOD == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.SOLVER.LR)
        self.test_accuracies = TestAccuracies([self.cfg.DATA.DATASET])
        
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.cfg.SOLVER.LR_SCH], gamma=0.1)
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.SOLVER.LR_SCH, gamma=0.9)
        
        self.start_iteration = 0
        if self.cfg.CHECKPOINT.RESUME_FROM_CHECKPOINT or self.cfg.TEST.ONLY_TEST:
            self.load_checkpoint()
        self.optimizer.zero_grad()

    def init_model(self):
        if self.cfg.MODEL.NAME == 'trx':
            from models.model_TRX import CNN_TRX as CNN
        elif self.cfg.MODEL.NAME == 'ta2n':
            from models.model_ta2n import CNN
        elif self.cfg.MODEL.NAME == 'strm':
            from models.model_strm import CNN_STRM as CNN
        elif self.cfg.MODEL.NAME == 'molo':
            from models.model_molo import CNN_BiMHM_MoLo as CNN
        elif self.cfg.MODEL.NAME == 'soap':
            from models.model_soap import CNN_SOAP as CNN
        elif self.cfg.MODEL.NAME == 'test':
            from models.model_test import CNN as CNN
        model = CNN(self.cfg)
        model = model.to(self.device)
        if self.cfg.DEVICE.NUM_GPUS > 1:
            model.distribute_model()

        print(f'inited model: {self.cfg.MODEL.NAME}\n')
        return model


    """
    Command line parser
    """
    def parse_config(self, cfg):
        

        print('learning rate decay scheduler', cfg.SOLVER.LR_SCH)

        if cfg.CHECKPOINT.CHECKPOINT_DIR == None:
            print("need to specify a checkpoint dir")
            exit(1)

        # if (args.backbone == "resnet50") or (args.backbone == "resnet34"):
        #     args.img_size = 224
        if cfg.MODEL.BACKBONE == "resnet50":
            cfg.trans_linear_in_dim = 2048
        else:
            cfg.trans_linear_in_dim = 512
        
        cfg.trans_linear_out_dim = cfg.MODEL.TRANS_LINEAR_OUT_DIM

        if cfg.DATA.DATASET == "ssv2":
            cfg.traintestlist = os.path.join("/home/zhangbin/tx/FSAR/splits/ssv2_OTAM")
            cfg.path = os.path.join(cfg.DATA.DATA_DIR, "ssv2_256x256q5_l8")
        if cfg.DATA.DATASET == 'ssv2_cmn':
            cfg.traintestlist = os.path.join("/home/zhangbin/tx/FSAR/splits/ssv2_CMN")
            cfg.path = os.path.join(cfg.DATA.DATA_DIR, "ssv2_256x256q5_l8")
        elif cfg.DATA.DATASET == 'hmdb':
            cfg.traintestlist = os.path.join("/home/sjtu/data/splits/hmdb_ARN/")
            cfg.path = os.path.join(cfg.DATA.DATA_DIR, "HMDB51/jpg")
        elif cfg.DATA.DATASET == 'ucf':
            cfg.traintestlist = os.path.join("/home/zhangbin/tx/FSAR/splits/ucf_ARN/")
            cfg.path = os.path.join(cfg.DATA.DATA_DIR, "ucf_256x256q5_l8")
        elif cfg.DATA.DATASET == 'kinetics':
            cfg.traintestlist = os.path.join("/home/sjtu/data/splits/kinetics_CMN/")
            cfg.path = os.path.join(cfg.DATA.DATA_DIR, "kinetics/Kinetics_frames")

        return cfg

    def run(self):
        if self.cfg.TEST.ONLY_TEST:
            print('Conduct Testing:')
            accuracy_dict = self.test()
            self.test_accuracies.print(self.logfile, accuracy_dict)
            print('Evaluation Done with', self.test_episodes, ' iteration')
        else:
            print('Conduct Training:')
            best_accuracies = 0.0
            train_accuracies = []
            losses = []
            total_iterations = self.train_episodes

            iteration = self.start_iteration
            for task_dict in self.video_loader:
                if iteration >= total_iterations:
                    break
                iteration += 1
                #print('iteration', iteration)
                torch.set_grad_enabled(True)
                task_loss, task_accuracy = self.train_task(task_dict)
                train_accuracies.append(task_accuracy)
                losses.append(task_loss)

                # optimize
                if ((iteration + 1) % self.cfg.TRAIN.TASKS_PER_BATCH == 0) or (iteration == (total_iterations - 1)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.scheduler.step()


                self.writer.add_scalar('loss/Train_loss[it]', task_loss, iteration + 1)
                self.writer.add_scalar('acc/Train_acc[it]', task_accuracy, iteration + 1)
                if (iteration + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                    # print training stats
                    print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                                    .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                            torch.Tensor(train_accuracies).mean().item()))
                    self.writer.add_scalar('loss/Train_loss[mean]', torch.Tensor(losses).mean().item(), (iteration + 1) // self.cfg.TRAIN.PRINT_FREQ)
                    self.writer.add_scalar('acc/Train_acc[mean]', torch.Tensor(train_accuracies).mean().item(), (iteration + 1) // self.cfg.TRAIN.PRINT_FREQ)
                    train_accuracies = []
                    losses = []

                if ((iteration + 1) % self.cfg.CHECKPOINT.SAVE_FREQ == 0) and (iteration + 1) != total_iterations:
                    #self.save_checkpoint(iteration + 1)
                    self.save_checkpoint(iteration + 1, 'last')


                if ((iteration + 1) % self.cfg.TRAIN.VAL_FREQ == 0) and (iteration + 1) != total_iterations:
                    accuracy_dict = self.test()
                    if accuracy_dict[self.cfg.DATA.DATASET]["accuracy"] > best_accuracies:
                        best_accuracies = accuracy_dict[self.cfg.DATA.DATASET]["accuracy"]
                        print('Save best checkpoint in {} iter'.format(iteration + 1))
                        self.save_checkpoint(iteration + 1, 'best')

                    self.writer.add_scalar('loss/Test_loss', accuracy_dict[self.cfg.DATA.DATASET]["loss"], (iteration + 1) // self.cfg.TRAIN.VAL_FREQ)
                    self.writer.add_scalar('acc/Test_acc', accuracy_dict[self.cfg.DATA.DATASET]["accuracy"], (iteration + 1) // self.cfg.TRAIN.VAL_FREQ)
                    self.writer.add_scalar('acc/Best_acc', best_accuracies, (iteration + 1) // self.cfg.TRAIN.VAL_FREQ)
                    self.test_accuracies.print(self.logfile, accuracy_dict)

        self.logfile.close()

    def train_task(self, task_dict):
        context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list, real_support_labels = self.prepare_task(task_dict)

        model_dict = self.model(context_images, context_labels, target_images)
        
        task_loss, task_acc = self._loss_and_acc(model_dict, target_labels, real_target_labels, batch_class_list, real_support_labels)
        task_loss.backward(retain_graph=False)

        return task_loss, task_acc

    def test(self):
        self.model.eval()
        with torch.no_grad():

                self.video_loader.dataset.train = False
                accuracy_dict ={}
                accuracies = []
                losses = []
                iteration = 0
                item = self.cfg.DATA.DATASET
                for task_dict in self.video_loader:
                    if iteration >= self.test_episodes:
                        break
                    iteration += 1

                    context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list, real_support_labels = self.prepare_task(task_dict)
                    model_dict = self.model(context_images, context_labels, target_images)

                    task_loss, task_acc = self._loss_and_acc(model_dict, target_labels, real_target_labels, batch_class_list, real_support_labels, mode='test')

                    losses.append(task_loss.item())
                    accuracies.append(task_acc.item())

                    current_accuracy = np.array(accuracies).mean() * 100.0
                    if self.cfg.TEST.ONLY_TEST:
                        self.writer.add_scalar(f'TEST/{self.cfg.DATA.DATASET}_{self.cfg.TRAIN.SHOT}-shot', current_accuracy, iteration+1)
                    print('current acc:{:0.3f} in iter:{:n}'.format(current_accuracy, iteration+1), end='\r',flush=True)

                accuracy = np.array(accuracies).mean() * 100.0
                loss = np.array(losses).mean()
                confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
                accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence, "loss": loss}
                self.video_loader.dataset.train = True
        self.model.train()
        
        return accuracy_dict


    def prepare_task(self, task_dict, images_to_device = True):
        context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
        target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]
        real_support_labels = task_dict['real_support_labels'][0]
        real_target_labels = task_dict['real_target_labels'][0]
        batch_class_list = task_dict['batch_class_list'][0]

        if images_to_device:
            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)
        real_target_labels = real_target_labels.to(self.device)
        real_support_labels = real_support_labels.to(self.device)

        return context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list, real_support_labels  

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]

    def _loss_and_acc(self, model_dict, target_labels, real_target_labels, batch_class_list, real_support_labels, mode='train'):
        lmd = 0.1
        model_dict = {k: v.to(self.device) for k,v in model_dict.items()}
        target_logits = model_dict['logits']

        if self.cfg.MODEL.NAME == 'strm':
            # Target logits after applying query-distance-based similarity metric on patch-level enriched features
            target_logits_post_pat = model_dict['logits_post_pat']

            # Add the logits before computing the accuracy
            target_logits = target_logits + lmd*target_logits_post_pat

            task_loss = self.loss(target_logits, target_labels, self.device) / self.cfg.TRAIN.TASKS_PER_BATCH
            task_loss_post_pat = self.loss(target_logits_post_pat, target_labels, self.device) / self.cfg.TRAIN.TASKS_PER_BATCH

            # Joint loss
            task_loss = task_loss + lmd*task_loss_post_pat
            task_accuracy = self.accuracy_fn(target_logits, target_labels)
            if mode == 'test':
                del target_logits
                del target_logits_post_pat
        elif self.cfg.MODEL.NAME == 'molo':
            if mode == 'test':
                task_loss = F.cross_entropy(model_dict["logits"], target_labels) /self.cfg.TRAIN.TASKS_PER_BATCH
                del target_logits
            else:
                task_loss =  (F.cross_entropy(model_dict["logits"], target_labels) \
                      + self.cfg.MODEL.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([real_support_labels, real_target_labels], 0).long())) /self.cfg.TRAIN.TASKS_PER_BATCH \
                        + self.cfg.MODEL.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q"], target_labels) /self.cfg.TRAIN.TASKS_PER_BATCH \
                            + self.cfg.MODEL.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s"], target_labels) /self.cfg.TRAIN.TASKS_PER_BATCH \
                                + self.cfg.MODEL.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q_motion"], target_labels) /self.cfg.TRAIN.TASKS_PER_BATCH \
                                    + self.cfg.MODEL.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s_motion"], target_labels) /self.cfg.TRAIN.TASKS_PER_BATCH \
                                        + self.cfg.MODEL.RECONS_COFF*model_dict["loss_recons"]
            task_accuracy = self.accuracy_fn(target_logits, target_labels)
        elif self.cfg.MODEL.NAME == 'soap':
            task_loss = self.loss(target_logits, target_labels, self.device) / self.cfg.TRAIN.TASKS_PER_BATCH + model_dict['t_loss']
            task_accuracy = self.accuracy_fn(target_logits, target_labels)
            if mode == 'test':
                del target_logits
        else:
            task_loss = self.loss(target_logits, target_labels, self.device) / self.cfg.TRAIN.TASKS_PER_BATCH
            task_accuracy = self.accuracy_fn(target_logits, target_labels)
            if mode == 'test':
                del target_logits
        
        return task_loss, task_accuracy

    def save_checkpoint(self, iteration, stat):
        d = {'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}

        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint_{}.pt'.format(stat)))
        #torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        if self.cfg.TEST.ONLY_TEST:
            checkpoint = torch.load(self.test_checkpoint_path, map_location=self.device)
            print(f'Load checkpoint from {self.test_checkpoint_path} ==> iter: [{checkpoint["iteration"]}]\n')
        else:
            checkpoint = torch.load(self.resume_checkpoint_path, map_location=self.device)
            print(f'Load checkpoint from {self.resume_checkpoint_path} ==> iter: [{checkpoint["iteration"]}]\n')
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

