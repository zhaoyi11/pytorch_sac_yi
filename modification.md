1. remove dmc2gym and change it to own implementation
2. change architecture of actor and critic to use layernorm
3. change the batch size from 1024 to 512
4. set tau from 0.99 per two update step to 0.995 per update step
5. set action repeat as 2