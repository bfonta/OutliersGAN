def add_args(parser):
    parser.add_argument(
        '--checkpoint',
        type=int,
        default=0,
        help='Checkpoint prefix number.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='Mode to run the model.'
    )
    return parser.parse_known_args()
