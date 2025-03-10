def set_train():    
    stock_symbol = '5871.TW'
    end_date = '2024-12-31'

    num_class = 2
    batch_size = 160
    init = True
    fp16_training = True
    num_epochs = 50
    lr = 0.01
    return stock_symbol, end_date, num_class, batch_size, init, fp16_training, num_epochs, lr