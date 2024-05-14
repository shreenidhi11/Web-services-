from pydantic import BaseModel, Field, ConfigDict
from bson import ObjectId
from typing import Optional, List
from typing_extensions import Annotated
from pydantic.functional_validators import BeforeValidator

PyObjectId = Annotated[str, BeforeValidator(str)]


class UserCredentailModel(BaseModel):
    """
    Container for a single order record.
    """

    # The primary key for the PurchaseOrderModel, stored as a `str` on the instance.
    # This will be aliased to `_id` when sent to MongoDB,
    # but provided as `id` in the API requests and responses.
    # id: Optional[PyObjectId] = Field(alias="_id", default=None)
    User_name: str = Field(...)
    User_pass: str = Field(...)
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "User_name": "Shreenidhi",
                "User_pass": "Shree1234",
            }
        },
    )

class UserCommunicationModel(BaseModel):
    User_name: str = Field(...)
    Operation_id : int
    model_config = ConfigDict(
        arbitrary_types_allowed = True,
        json_encoders = {ObjectId: str},
        json_schema_extra = {
             "example": {
                "User_name": "Shreenidhi",
                "Operation_id": 1,
            }
        }
    )

class LaptopDetails(BaseModel):
    # _id: str = Field(..., alias="_id")
    Sequential_ID: int
    Company: str
    TypeName: str
    Inches: float
    ScreenResolution: str
    Cpu: str
    Memory: str
    Gpu: str
    OpSys: str
    Price: float
    Ram_In_GB: int
    Weight_in_kg: float
    CPU_GZ: float
    CPU_Gen: str


