def test(model, optimizer, mnist_test, epoch, best_test_loss):
    test_avg_loss = 0.0
    with torch.no_grad():  # 这一部分不计算梯度，也就是不放入计算图中去
        '''测试测试集中的数据'''
        # 计算所有batch的损失函数的和
        for test_batch_index, (test_x, _) in enumerate(mnist_test):
            test_x = test_x.to(device)
            # 前向传播
            test_x_hat, test_mu, test_log_var = model(test_x)
            # 损害函数值
            test_loss, test_BCE, test_KLD = utils.get_loss(test_x_hat, test_x, test_mu, test_log_var)
            test_avg_loss += test_loss

        # 对和求平均，得到每一张图片的平均损失
        test_avg_loss /= len(mnist_test.dataset)

        '''测试随机生成的隐变量'''
        # 随机从隐变量的分布中取隐变量
        z = torch.randn(hparams.batch_size, hparams.z_dim).to(device)  # 每一行是一个隐变量，总共有batch_size行
        # 对隐变量重构
        random_res = model.decode(z).view(-1, 1, 28, 28)
        # 保存重构结果
        save_image(random_res, './%s/random_sampled-%d.png' % (hparams.result_dir, epoch + 1))

        '''保存目前训练好的模型'''
        # 保存模型
        is_best = test_avg_loss < best_test_loss
        best_test_loss = min(test_avg_loss, best_test_loss)
        utils.save_checkpoint({
            'epoch': epoch,  # 迭代次数
            'best_test_loss': best_test_loss,  # 目前最佳的损失函数值
            'state_dict': model.state_dict(),  # 当前训练过的模型的参数
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_dir)

        return best_test_loss
