# Assumption

- Warehouse -> rectangular

- Receiption only on 1 side -> receiving completed -> picking

- Each warehouse / 2 area: carton & plastic


# Objective --> minimize distance travel in the warehouse ! Take into account the stores shipping 
                prioties 

distance ______> picking
        |______> shipping

shipping distance -- mỗi đơn hàng của một store đc vận chuyển từ floor location đến shipping point
                     một cách độc lập

                  -- hàng của mỗi store có nhiều pallet và được vận chuyển hết bằng một hoặc nhiều lần
                     vận chuyển

picking distance

cluster of floor location : một nhóm N floor sát nhau, mỗi pick up route sẽ là khoảng cách tính từ floor
                            đến điểm nhận và từ điểm nhận vể floor đó ==> tất cả floor trong cùng 1 cluster 
                            sẽ có cùng 1 picking distance

==> Giả thiết phải có số lượng floor/ cluster và số lượng cluster

Dữ liệu đầu vào cần:
- danh sách store có nhu cầu hàng và qty hàng mỗi store
- danh sách vị trí floor 
- danh sách từng floor đến receiption point
- danh sách từng floor đến shipping point
- 
